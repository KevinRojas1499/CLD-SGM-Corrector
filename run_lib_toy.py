# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License for
# CLD-SGM. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import shutil
import os
import time
import logging
import plotly
import torch
from torch.utils import tensorboard
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from matplotlib import cm

from models import mlp, gmm
from models.ema import ExponentialMovingAverage
from models import utils as mutils
from util.utils import make_dir, get_optimizer, optimization_manager, set_seeds, compute_eval_loss, compute_non_image_likelihood, broadcast_params, reduce_tensor, build_beta_fn, build_beta_int_fn
from util.checkpoint import save_checkpoint, restore_checkpoint
import losses
import sde_lib
import sampling
import likelihood
from util.toy_data import inf_data_gen

import wandb
import pandas as pd

def train(config, workdir):
    ''' Main training script. '''

    local_rank = config.local_rank
    global_rank = config.global_rank
    global_size = config.global_size

    if config.mode == 'train':
        set_seeds(global_rank, config.seed)
    elif config.mode == 'continue':
        set_seeds(global_rank, config.seed + config.cont_nbr)
    else:
        raise NotImplementedError('Mode %s is unknown.' % config.mode)

    torch.cuda.device(local_rank)
    config.device = torch.device('cuda:%d' % local_rank)

    # Setting up all necessary folders
    sample_dir = os.path.join(workdir, 'samples')
    tb_dir = os.path.join(workdir, 'tensorboard')
    checkpoint_dir = os.path.join(workdir, 'checkpoints')
    likelihood_dir = os.path.join(workdir, 'likelihood')

    if global_rank == 0:
        logging.info(config)
        if config.mode == 'train':
            make_dir(sample_dir)
            make_dir(tb_dir)
            make_dir(checkpoint_dir)
            make_dir(likelihood_dir)
        writer = tensorboard.SummaryWriter(tb_dir)
    dist.barrier()

    beta_fn = build_beta_fn(config)
    beta_int_fn = build_beta_int_fn(config)

    if config.sde == 'vpsde':
        sde = sde_lib.VPSDE(config, beta_fn, beta_int_fn)
    elif config.sde == 'cld':
        sde = sde_lib.CLD(config, beta_fn, beta_int_fn)
    else:
        raise NotImplementedError('SDE %s is unknown.' % config.sde)

    # Creating the score model
    score_model = mutils.create_model(config).to(config.device)
    broadcast_params(score_model.parameters())  # Sync all parameters
    score_model = DDP(score_model, device_ids=[local_rank])

    ema = ExponentialMovingAverage(
        score_model.parameters(), decay=config.ema_rate)

    if global_rank == 0:
        model_parameters = filter(
            lambda p: p.requires_grad, score_model.parameters())
        n_params = sum([np.prod(p.size()) for p in model_parameters])
        logging.info('Number of trainable parameters in model: %d' % n_params)
    dist.barrier()

    optim_params = score_model.parameters()
    optimizer = get_optimizer(config, optim_params)
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

    if config.mode == 'continue':
        if config.checkpoint is None:
            ckpt_path = os.path.join(checkpoint_dir, 'checkpoint.pth')
        else:
            ckpt_path = os.path.join(checkpoint_dir, config.checkpoint)

        if global_rank == 0:
            logging.info('Loading model from path: %s' % ckpt_path)
        dist.barrier()

        state = restore_checkpoint(ckpt_path, state, device=config.device)

    num_total_iter = config.n_train_iters

    if global_rank == 0:
        logging.info('Number of total iterations: %d' % num_total_iter)
    dist.barrier()

    optimize_fn = optimization_manager(config)
    train_step_fn = losses.get_step_fn(True, optimize_fn, sde, config)

    sampling_shape = (config.sampling_batch_size,
                      config.data_dim)
    sampling_fn = sampling.get_sampling_fn(
        config, sde, sampling_shape, config.sampling_eps)

    likelihood_fn = likelihood.get_likelihood_fn(config, sde)

    step = int(state['step'])
    if global_rank == 0:
        logging.info('Starting training at step %d' % step)
    dist.barrier()

    if config.mode == 'continue':
        config.eval_threshold = max(step + 1, config.eval_threshold)
        config.snapshot_threshold = max(step + 1, config.snapshot_threshold)
        config.likelihood_threshold = max(
            step + 1, config.likelihood_threshold)
        config.save_threshold = max(step + 1, config.save_threshold)

    while step < num_total_iter:
        if step % config.likelihood_freq == 0 and step >= config.likelihood_threshold:
            ema.store(score_model.parameters())
            ema.copy_to(score_model.parameters())
            mean_nll = compute_non_image_likelihood(
                config, sde, state, likelihood_fn, inf_data_gen, step=step, likelihood_dir=likelihood_dir)
            ema.restore(score_model.parameters())

            if global_rank == 0:
                logging.info('Mean Nll at step: %d: %.5f' %
                             (step, mean_nll.item()))
                writer.add_scalar('mean_nll', mean_nll.item(), step)

                checkpoint_file = os.path.join(
                    checkpoint_dir, 'checkpoint_%d.pth' % step)
                if not os.path.isfile(checkpoint_file):
                    save_checkpoint(checkpoint_file, state)
            dist.barrier()

        if (step % config.snapshot_freq == 0 or step == num_total_iter) and global_rank == 0 and step >= config.snapshot_threshold:
            logging.info('Saving snapshot checkpoint.')
            save_checkpoint(os.path.join(
                checkpoint_dir, 'snapshot_checkpoint.pth'), state)

            ema.store(score_model.parameters())
            ema.copy_to(score_model.parameters())
            x, v, nfe = sampling_fn(score_model)
            ema.restore(score_model.parameters())

            logging.info('NFE snapshot at step %d: %d' % (step, nfe))
            writer.add_scalar('nfe', nfe, step)

            this_sample_dir = os.path.join(sample_dir, 'iter_%d' % step)
            make_dir(this_sample_dir)

            plt.scatter(x.cpu().numpy()[:, 0], x.cpu().numpy()[:, 1], s=3)
            plt.savefig(os.path.join(this_sample_dir,
                        'sample_rank_%d.png' % global_rank))
            plt.close()

            if config.sde == 'cld':
                np.save(os.path.join(this_sample_dir, 'sample_x'), x.cpu())
                np.save(os.path.join(this_sample_dir, 'sample_v'), v.cpu())
            else:
                np.save(os.path.join(this_sample_dir, 'sample'), x.cpu())
        dist.barrier()

        if config.save_freq is not None:
            if step % config.save_freq == 0 and step >= config.save_threshold:
                if global_rank == 0:
                    checkpoint_file = os.path.join(
                        checkpoint_dir, 'checkpoint_%d.pth' % step)
                    if not os.path.isfile(checkpoint_file):
                        save_checkpoint(checkpoint_file, state)
                dist.barrier()

        # Training
        start_time = time.time()

        x = inf_data_gen(config.dataset, config.training_batch_size).to(
            config.device)
        loss = train_step_fn(state, x)

        if step % config.log_freq == 0:
            loss = reduce_tensor(loss, global_size)
            if global_rank == 0:
                logging.info('Iter %d/%d Loss: %.4f Time: %.3f' % (step + 1,
                             config.n_train_iters, loss.item(), time.time() - start_time))
                writer.add_scalar('training_loss', loss, step)
            dist.barrier()

        step += 1

    if global_rank == 0:
        logging.info('Finished after %d iterations.' % config.n_train_iters)
        logging.info('Saving final checkpoint.')
        save_checkpoint(os.path.join(
            checkpoint_dir, 'final_checkpoint.pth'), state)
    dist.barrier()


def get_run_name(config):
    file_name = file_name= f"{config.sampling_method}_{config.n_discrete_steps}"
    if config.sampling_method == 'corrector':
        file_name+= f"_{config.predictor_fast_steps}_{config.eta : .3f}_lang_{config.n_lang_iters}"

    return file_name

def init_wandb(config):
    run_name = get_run_name(config)
    wandb.init(
        # set the wandb project where this run will be logged
        project=config.wandb_project,
        name= run_name,
        # track hyperparameters and run metadata
        config=config
    )

def create_relevant_config(config):
    small_config = {
        "sampling_method": config.sampling_method,
        "sampling_eps": config.sampling_eps,
        "denoising": config.denoising,
        "n_discrete_steps": config.n_discrete_steps,
        "n_lang_iters": config.n_lang_iters,
        "h_lang": config.h_lang,
        "overdamped_lang": config.overdamped_lang,
        "langevin_friction": config.langevin_friction,
        "micro_eps": config.micro_eps,
        "eta": config.eta,
        "predictor_fast_steps": config.predictor_fast_steps,
        "correct": config.correct
    }
    return small_config

def compute_stats_gmm(data, config):
    limit = 3
    bias = config.mean
    clusters = [ [] for i in range(5)]
    # center, up right, up left , down left, down right
    for point in data:
        x,y = point
        if x > limit + bias and y > limit + bias:
            clusters[1].append(point)
        elif x < -limit + bias and y > limit + bias:
            clusters[2].append(point)
        elif x < -limit + bias and y < -limit + bias:
            clusters[3].append(point)
        elif x > limit + bias and y < -limit + bias:
            clusters[4].append(point)
        else:
            clusters[0].append(point)

    stats_x = {"center":0,"up right":0, "up left":0, "down left": 0, "down right":0}
    stats_y = {"center":0,"up right":0, "up left":0, "down left": 0, "down right":0}
    weights = {"center":0,"up right":0, "up left":0, "down left": 0, "down right":0}

    for i, (key, value) in enumerate((stats_x.items())):
        mean = np.mean(np.array(clusters[i]),axis=0) 
        stats_x[key] = mean[0]
        stats_y[key] = mean[1]
        weights[key] = len(clusters[i])/len(data)
    
    return stats_x, stats_y, weights

def to_np_array(data):
    np_data = np.zeros(len(data))
    for i, (key, value) in enumerate((data.items())):
        np_data[i] = value
    
    return np_data

def summarized_stats(data, config):
    stats_x, stats_y, weights = compute_stats_gmm(data, config)
    weights = to_np_array(weights)
    real_weights = np.array([.2,.2,.2,.2,.2])
    w = np.sum((weights-real_weights)**2)**.5

    means_x, means_y = np.expand_dims(to_np_array(stats_x),axis=1), np.expand_dims(to_np_array(stats_y),axis=1)
    m = config.mean
    b = config.intercept
    real_means = np.array([[m,m],[m+b,m+b],[m-b,m+b],[m-b, m -b],[m + b,m - b]])
    means = np.concatenate((means_x,means_y), axis=1)
    error_means = 0
    for i, mean in enumerate(real_means):
        error_means += np.sum((real_means[i]-means[i])**2)**.5

    return w, error_means


def evaluate(config, workdir):
    ''' Main evaluation script. '''

    local_rank = config.local_rank
    global_rank = config.global_rank
    set_seeds(global_rank, config.seed + config.eval_seed)

    torch.cuda.device(local_rank)
    config.device = torch.device('cuda:%d' % local_rank)

    checkpoint_dir = os.path.join(workdir, 'checkpoints')
    eval_dir = os.path.join(workdir, config.eval_folder)
    sample_dir = os.path.join(eval_dir, 'samples')
    if global_rank == 0:
        logging.info(config)
        make_dir(sample_dir)
    dist.barrier()

    beta_fn = build_beta_fn(config)
    beta_int_fn = build_beta_int_fn(config)

    if config.sde == 'vpsde':
        sde = sde_lib.VPSDE(config, beta_fn, beta_int_fn)
    elif config.sde == 'cld':
        sde = sde_lib.CLD(config, beta_fn, beta_int_fn)
    else:
        raise NotImplementedError('SDE %s is unknown.' % config.vpsde)

    score_model = mutils.create_model(config).to(config.device)
    if config.name != 'gmm':
        broadcast_params(score_model.parameters())
        score_model = DDP(score_model, device_ids=[local_rank])
        ema = ExponentialMovingAverage(
            score_model.parameters(), decay=config.ema_rate)

        optim_params = score_model.parameters()
        optimizer = get_optimizer(config, optim_params)
        state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

        optimize_fn = optimization_manager(config)
        eval_step_fn = losses.get_step_fn(False, optimize_fn, sde, config)

        ckpt_path = os.path.join(checkpoint_dir, config.ckpt_file)
        state = restore_checkpoint(ckpt_path, state, device=config.device)
        ema.copy_to(score_model.parameters())

        step = int(state['step'])
        if global_rank == 0:
            logging.info('Evaluating at training step %d' % step)
        dist.barrier()

    sampling_shape = (config.sampling_batch_size,
                      config.data_dim)
    sampling_fn = sampling.get_sampling_fn(
        config, sde, sampling_shape, config.sampling_eps)

    likelihood_fn = likelihood.get_likelihood_fn(config, sde)




    if config.eval_loss:
        eval_loss = compute_eval_loss(
            config, sde, state, eval_step_fn, inf_data_gen)
        if global_rank == 0:
            logging.info("Testing loss: %.5f" % eval_loss.item())
        dist.barrier()

    if config.eval_likelihood:
        mean_nll = compute_non_image_likelihood(
            config, sde, state, likelihood_fn, inf_data_gen)
        if global_rank == 0:
            logging.info("Mean N-6.838L: %.5f" % mean_nll.item())
        dist.barrier()

    if config.eval_sample:
        relevant_config = create_relevant_config(config)
        x, _, nfe = sampling_fn(score_model)

        logging.info('NFE: %d' % nfe)
        print("Number of nan elements ",torch.sum(torch.isnan(x)))

        x = x.cpu().numpy()

        if config.mode == 'summary':
            return x

        stats_x, stats_y, weights = compute_stats_gmm(x,config)
        combined_stats = {}
        for key, value in stats_x.items():
            combined_stats[key] = weights[key], stats_x[key], stats_y[key]
        stats_df = pd.DataFrame(combined_stats, index=["Weights","Mean x", "Mean y"])
        stats_tbl = wandb.Table(data=stats_df)
        config_df = pd.DataFrame(relevant_config,index=[0])
        config_tbl = wandb.Table(data=config_df)
        
        init_wandb(config)

        table = wandb.Table(data=x,columns=["x","y"])
        wandb.log({"Generated Samples" : wandb.plot.scatter(table,"x","y")})
        wandb.log({"Summary statistics " : stats_tbl, "Config" : config_tbl})

        wandb.finish()

def reset_config(config):
    config.n_discrete_steps = 40
    config.n_lang_iters = 5
    config.skip_predictor = False
    config.sampling_method = 'corrector'


def summarize(config, workdir):
    init_wandb(config)

    reset_config(config)
    # Summarize disc steps
    summarize_fixed_var(config, workdir, 'number of disc steps', config.n_discrete_steps_range)
    print("Finished disc steps")
    reset_config(config)

    # # Summarize corrector steps
    # summarize_fixed_var(config, workdir, 'number of corrector steps', config.n_lang_iters_range)
    # print("Finished corrector steps")
    # reset_config(config)

    # # Summarize means steps
    # summarize_fixed_var(config, workdir, 'mean of distribution', config.means_range)
    # print("Finished mean")

    wandb.finish()
    return 0

def summarize_fixed_var(config, workdir, variable, possible_values):
    weights_stats = []
    means_stats  = []
    weights_stats_no_pred = []
    means_stats_no_pred  = []
    weights_stats_em = []
    means_stats_em  = []
    for val in possible_values:
        reset_config(config)

        if variable == 'number of disc steps':
            config.n_discrete_steps = val
        elif variable == 'number of corrector steps':
            config.n_lang_iters = val
        elif variable == 'mean of distribution':
            config.mean = val
        x = evaluate(config,workdir)
        weights_error, means_error = summarized_stats(x, config)

        weights_stats.append(weights_error)
        means_stats.append(means_error)

        shutil.rmtree(os.path.join(workdir,"eval_fid"))

        config.skip_predictor = True

        x = evaluate(config,workdir)
        weights_error, means_error = summarized_stats(x, config)

        weights_stats_no_pred.append(weights_error)
        means_stats_no_pred.append(means_error)

        shutil.rmtree(os.path.join(workdir,"eval_fid"))

        if variable == 'number of disc steps':
            config.skip_predictor = False
            config.sampling_method = 'em'
            config.n_discrete_steps = val * config.n_lang_iters

            x = evaluate(config,workdir)
            weights_error, means_error = summarized_stats(x, config)

            weights_stats_em.append(weights_error)
            means_stats_em.append(means_error)

            shutil.rmtree(os.path.join(workdir,"eval_fid"))
    
    weights_fig = go.Figure()
    weights_fig.add_trace(go.Scatter(x=possible_values, y = weights_stats, name="Predictor"))
    weights_fig.add_trace(go.Scatter(x=possible_values, y = weights_stats_no_pred, name="No Predictor"))
    if variable == 'number of disc steps':
        weights_fig.add_trace(go.Scatter(x=possible_values, y = weights_stats_em, name="Euler Maruyama"))

    title_weights = f"L2 Error in Weights as a function of {variable}"
    weights_fig.update_layout(title=title_weights,
                          xaxis_title=f"{variable}",
                          yaxis_title="Error")
    means_fig = go.Figure()
    means_fig.add_trace(go.Scatter(x=possible_values, y = means_stats, name="Predictor"))
    means_fig.add_trace(go.Scatter(x=possible_values, y = means_stats_no_pred, name="No Predictor"))
    if variable == 'number of disc steps':
        means_fig.add_trace(go.Scatter(x=possible_values, y = means_stats_em, name="Euler - Maruyama"))

    title_means = f"Sum of L2 Error in Means as a function of {variable}"
    means_fig.update_layout(title=title_means,
                          xaxis_title=f"{variable}",
                          yaxis_title="Error")

    wandb.log({title_weights: weights_fig, title_means: means_fig})