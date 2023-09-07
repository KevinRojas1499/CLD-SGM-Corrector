# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License for
# CLD-SGM. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import torch
import gc
from torchdiffeq import odeint
from models.utils import get_score_fn
from tqdm import tqdm

def get_sampling_fn(config, sde, sampling_shape, eps):
    sampler_name = config.sampling_method
    if sampler_name == 'ode':
        return get_ode_sampler(config, sde, sampling_shape, eps)
    elif sampler_name == 'em':
        return get_em_sampler(config, sde, sampling_shape, eps)
    elif sampler_name == 'sscs':
        return get_sscs_sampler(config, sde, sampling_shape, eps)
    elif sampler_name == 'corrector':
        return get_corrector_sampler(config,sde, sampling_shape, eps)
    else:
        raise NotImplementedError(
            'Sampler %s is not implemened.' % sampler_name)


def get_ode_sampler(config, sde, sampling_shape, eps):
    ''' 
    Sampling from ProbabilityFlow formulation. 
    '''
    gc.collect()

    def denoising_fn(model, u, t):
        score_fn = get_score_fn(config, sde, model, train=False)
        discrete_step_fn = sde.get_discrete_step_fn(
            mode='reverse', score_fn=score_fn)
        u, u_mean = discrete_step_fn(u, t, eps)
        return u_mean

    def probability_flow_ode(model, u, t):
        ''' 
        The "Right-Hand Side" of the ODE. 
        '''
        score_fn = get_score_fn(config, sde, model, train=False)
        rsde = sde.get_reverse_sde(score_fn, probability_flow=True)
        return rsde(u, t)[0]

    def ode_sampler(model, u=None):
        with torch.no_grad():
            if u is None:
                x, v = sde.prior_sampling(sampling_shape)
                if sde.is_augmented:
                    u = torch.cat((x, v), dim=1)
                else:
                    u = x

            def ode_func(t, u):
                global nfe_counter
                nfe_counter += 1
                vec_t = torch.ones(
                    sampling_shape[0], device=u.device, dtype=torch.float64) * t
                dudt = probability_flow_ode(model, u, vec_t)
                return dudt

            global nfe_counter
            nfe_counter = 0
            time_tensor = torch.tensor(
                [0., 1. - eps], dtype=torch.float64, device=config.device)
            solution = odeint(ode_func,
                              u,
                              time_tensor,
                              rtol=config.sampling_rtol,
                              atol=config.sampling_atol,
                              method=config.sampling_solver,
                              options=config.sampling_solver_options)

            u = solution[-1]

            if config.denoising:
                u = denoising_fn(model, u, 1. - eps)
                nfe_counter += 1

            if sde.is_augmented:
                x, v = torch.chunk(u, 2, dim=1)
                return x, v, nfe_counter
            else:
                return u, None, nfe_counter

    return ode_sampler

def get_corrector_sampler(config, sde, sampling_shape, sampling_eps):

    gc.collect()

    t_final = 1. - sampling_eps
    n_discrete_steps = config.n_discrete_steps

    beta = sde.beta_fn(0)
    m_inv = sde.m_inv
    gamma = 2/m_inv**.5
    # Discrete stuff
    eps = config.micro_eps
    predictor_fast_steps = config.predictor_fast_steps
    eta = t_final/n_discrete_steps/predictor_fast_steps


    # Langevin parameters    
    n_lang_iters = config.n_lang_iters
    h_lang = config.h_lang
    langevin_friction = config.langevin_friction
    
    c_hat = 2 * (gamma/(eta * beta)) **.5
    print(f"CHAT {c_hat}, ETA {eta}")

    # Plotting
    counter = 0

    def make_image(u, color='blue',name=None):
        import matplotlib.pyplot as plt
        import os
        nonlocal counter
        file_name= f"{counter}.png" if name == None else name
        counter+=1
        l = 13
        bias = 7
        plt.xlim(-l + bias,l + bias)
        plt.ylim(-l + bias,l + bias)
        x, _ = torch.chunk( u , 2, dim = 1)
        plt.scatter(x.cpu().numpy()[:, 0], x.cpu().numpy()[:, 1], color=color, s=3)
        plt.savefig(os.path.join("./root/trajectory/", file_name))
        plt.close()
        # print(torch.sum(torch.isnan(x)))

    def step_fn(model, u, t, dt):
        score_fn = get_score_fn(config, sde, model, train=False)
        discrete_step_fn = sde.get_discrete_step_fn(
            mode='reverse', score_fn=score_fn)
        u, u_mean = discrete_step_fn(u, t, dt)
        return u, u_mean

    def get_edm_discretization(num, device):
                rho=7
                sigma_min = 0.002
                step_indices = torch.arange(num, dtype=torch.float64, device=device)
                t_steps = (1 ** (1 / rho) + step_indices / (num - 1) * (sigma_min ** (1 / rho) - 1 ** (1 / rho))) ** rho
                t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0
                return t_steps

    def gradV(x):
        f1 = lambda x : torch.sin(x)
        return x + c_hat *  f1(x/eps)
        
    def fast_discrete_step_fn(u,t,dt):
        beta = sde.beta_fn(t)
        x,v = torch.chunk(u, 2, dim=1)
        dt = -dt
        v = (v + beta * gradV(x) * dt)/(1-4 / gamma  * beta * dt)
        x = x - 4/gamma**2 * beta * v * dt
        return torch.cat((x,v), dim=1)


    def discrete_steps(u, t , dt):
        t = torch.linspace(t + dt, t, predictor_fast_steps + 1, dtype=torch.float64)
        for i in range(predictor_fast_steps):
            dt = t[i+1]- t[i]
            u = fast_discrete_step_fn(u,t[i],dt)
        return u
    
    def overdamped_langevin_iter(u, h, potential):
        # Notice that the potential has a +, correctly it is a - but since the score needs a - to be the right potential I fix it here
        return u + potential*h + (2*h)**.5 * torch.randn_like(u)
    
    def underdamped_langevin_iter(u, v, h, potential):
        # Notice that the potential has a +, correctly it is a - but since the score needs a - to be the right potential I fix it here
        u = u + v * h
        v = v + (potential - langevin_friction * v)*h + (2*h*langevin_friction )**.5 * torch.randn_like(u)
        return u,v
    
    
    def overdamped_langevin_corrector(model, u, t, plot=False):
        tt = torch.ones(
            u.shape[0], device=u.device, dtype=torch.float64) * t
        
        score_fn = get_score_fn(config, sde, model, train=False)
        for i in range(n_lang_iters):
            score = score_fn(u,1. - tt)
            x,v = torch.chunk(u, 2, dim=1)
            sx,sv = torch.chunk(score, 2, dim=1)
            if score.shape[-1] == v.shape[-1]:
                sv = score
            if config.correct == 'both':
                x = overdamped_langevin_iter(x,h_lang,sx)
                v = overdamped_langevin_iter(v,h_lang,sv)
            elif config.correct == 'speed':
                v = overdamped_langevin_iter(v,h_lang,sv)
            else:
                x = overdamped_langevin_iter(x,h_lang,sx)
            u = torch.cat((x,v), dim=1)
        return u

    def underdamped_langevin_corrector(model, u, t):
        tt = torch.ones(u.shape[0], device=u.device, dtype=torch.float64) * t

        score_fn = get_score_fn(config, sde, model, train=False)
        v = torch.randn_like(u)
        for i in range(n_lang_iters):
            score = score_fn(u,1. - tt)
            if config.correct == 'both':
                u, v = underdamped_langevin_iter(u,v,h_lang,score)
        return u 
    

    def corrector_sampler(model, u=None):
        plot = config.plot_trajectory
        with torch.no_grad():
            if u is None:
                x, v = sde.prior_sampling(sampling_shape)
                if sde.is_augmented:
                    u = torch.cat((x, v), dim=1)
                else:
                    u = x


            print("Final time " , t_final)
            effective_step_size = predictor_fast_steps * eta
            # Notice that every predictor steps spans a range of predictor_fast_steps * eta

            print(n_discrete_steps)
            t = torch.linspace(
                0, t_final,  n_discrete_steps + 1, dtype=torch.float64)
            if config.striding == 'linear':
                pass
            elif config.striding == 'quadratic':
                t = t_final * torch.flip(1 - (t / t_final) ** 2., dims=[0])

            t = 0
            plot = config.plot_trajectory
            bar = tqdm(range(n_discrete_steps - 1))
            for i in bar:
                u = discrete_steps(u, t, effective_step_size)
                # u, _ = step_fn(model, u, t[i], dt)
                if plot:
                    make_image(u,color='blue',name=f"{i}_{t:.3f}.png")

                t+=effective_step_size
                bar.set_description(f"T : {t : .3f}")
                if config.overdamped_lang:
                    u = overdamped_langevin_corrector(model, u, t,plot=plot)
                else:
                    u = underdamped_langevin_corrector(model, u, t)
                if plot:
                    make_image(u,color='red',name=f"{i}_{t:.3f}.png")

            u, _ = step_fn(model, u, t, t_final - t)

            if config.denoising:
                _, u = step_fn(model, u, 1. - eps, eps)

            if sde.is_augmented:
                x, v = torch.chunk(u, 2, dim=1)
                return x, v, config.n_discrete_steps
            else:
                return u, None, config.n_discrete_steps
    
    return corrector_sampler

def get_em_sampler(config, sde, sampling_shape, eps):
    ''' 
    Sampling from the ReverseSDE using Euler--Maruyama. 
    '''

    gc.collect()

    def step_fn(model, u, t, dt):
        score_fn = get_score_fn(config, sde, model, train=False)
        discrete_step_fn = sde.get_discrete_step_fn(
            mode='reverse', score_fn=score_fn)
        u, u_mean = discrete_step_fn(u, t, dt)
        return u, u_mean

    def em_sampler(model, u=None):
        with torch.no_grad():
            if u is None:
                x, v = sde.prior_sampling(sampling_shape)
                if sde.is_augmented:
                    u = torch.cat((x, v), dim=1)
                else:
                    u = x

            n_discrete_steps = config.n_discrete_steps if not config.denoising else config.n_discrete_steps - 1
            t_final = 1. - eps
            t = torch.linspace(
                0., t_final, n_discrete_steps + 1, dtype=torch.float64)
            if config.striding == 'linear':
                pass
            elif config.striding == 'quadratic':
                t = t_final * torch.flip(1 - (t / t_final) ** 2., dims=[0])

            for i in range(n_discrete_steps):
                dt = t[i + 1] - t[i]
                u, _ = step_fn(model, u, t[i], dt)

            if config.denoising:
                _, u = step_fn(model, u, 1. - eps, eps)

            if sde.is_augmented:
                x, v = torch.chunk(u, 2, dim=1)
                return x, v, config.n_discrete_steps
            else:
                return u, None, config.n_discrete_steps

    return em_sampler


def get_sscs_sampler(config, sde, sampling_shape, eps):
    ''' 
    Sampling from the ReverseSDE using our SSCS. Only applicable to CLD-SGM.
    '''

    gc.collect()

    n_discrete_steps = config.n_discrete_steps if not config.denoising else config.n_discrete_steps - 1
    t_final = 1. - eps
    t = torch.linspace(0., t_final, n_discrete_steps + 1, dtype=torch.float64)
    if config.striding == 'linear':
        pass
    elif config.striding == 'quadratic':
        t = t_final * torch.flip(1 - (t / t_final) ** 2., dims=[0])

    beta_fn = sde.beta_fn
    beta_int_fn = sde.beta_int_fn
    num_stab = config.sscs_num_stab

    def denoising_fn(model, u, t):
        score_fn = get_score_fn(config, sde, model, train=False)
        discrete_step_fn = sde.get_discrete_step_fn(
            mode='reverse', score_fn=score_fn)
        u, u_mean = discrete_step_fn(u, t, eps)
        return u_mean

    def compute_mean_of_analytical_dynamics(u, t, dt):
        B = (beta_int_fn(1. - (t + dt)) - beta_int_fn(1. - t))

        x, v = torch.chunk(u, 2, dim=1)
        coeff = torch.exp(2. * sde.g * B)

        mean_x = coeff * ((1. - 2. * sde.g * B) * x + 4. * sde.g ** 2. * B * v)
        mean_v = coeff * (-B * x + (1. + 2. * sde.g * B) * v)
        return torch.cat((mean_x, mean_v), dim=1)

    def compute_variance_of_analytical_dynamics(t, dt):
        B = beta_int_fn(1. - (t + dt)) - beta_int_fn(1. - t)
        coeff = torch.exp(4. * sde.g * B)
        var_xx = coeff * (1. / coeff - 1. + 4. * sde.g *
                          B - 8. * sde.g**2 * B ** 2.)
        var_xv = -coeff * (4. * sde.g * B ** 2.)
        var_vv = coeff * (-sde.f ** 2. * (-(1. / coeff) +
                          1.) / 4. - sde.f * B - 2. * B ** 2.)

        return [var_xx + num_stab, var_xv, var_vv + num_stab]

    def analytical_dynamics(u, t, dt, half_step):
        if half_step:
            dt_hd = dt / 2.
        else:
            dt_hd = dt

        mean = compute_mean_of_analytical_dynamics(u, t, dt_hd)
        var = compute_variance_of_analytical_dynamics(t, dt_hd)

        cholesky11 = (torch.sqrt(var[0]))
        cholesky21 = (var[1] / cholesky11)
        cholesky22 = (torch.sqrt(var[2] - cholesky21 ** 2.))

        if torch.sum(torch.isnan(cholesky11)) > 0 or torch.sum(torch.isnan(cholesky22)) > 0:
            raise ValueError('Numerical precision error.')

        batch_randn = torch.randn_like(u, device=u.device)
        batch_randn_x, batch_randn_v = torch.chunk(batch_randn, 2, dim=1)

        noise_x = cholesky11 * batch_randn_x
        noise_v = cholesky21 * batch_randn_x + cholesky22 * batch_randn_v
        noise = torch.cat((noise_x, noise_v), dim=1)

        mean = mean
        noise = noise
        perturbed_data = mean + noise
        return perturbed_data

    def euler_score_dynamics(model, u, t, dt, half_step):
        if half_step:
            raise ValueError('Avoid half steps in score dynamics.')

        score_fn = get_score_fn(config, sde, model, train=False)
        score = score_fn(u, torch.ones(
            u.shape[0], device=u.device, dtype=torch.float64) * (1. - t))

        x, v = torch.chunk(u, 2, dim=1)
        v_new = v + 2. * sde.f * (score + sde.m_inv * v) * beta_fn(1. - t) * dt

        return torch.cat((x, v_new), dim=1)

    def sscs_sampler(model, u=None):
        ''' 
        The SSCS sampler takes analytical "half-steps" for the Ornstein--Uhlenbeck
        and the Hamiltonian components, and evaluates the score model using "full-steps". 
        '''

        with torch.no_grad():
            if u is None:
                x, v = sde.prior_sampling(sampling_shape)
                if sde.is_augmented:
                    u = torch.cat((x, v), dim=1)
                else:
                    raise ValueError('SSCS sampler does only work for CLD.')
            else:
                if not sde.is_augmented:
                    raise ValueError('SSCS sampler does only work for CLD.')

            for i in range(n_discrete_steps):
                dt = t[i + 1] - t[i]
                u = analytical_dynamics(u, t[i], dt, True)
                u = euler_score_dynamics(model, u, t[i], dt, False)
                u = analytical_dynamics(u, t[i], dt, True)

            if config.denoising:
                u = denoising_fn(model, u, 1.0 - eps)

            x, v = torch.chunk(u, 2, dim=1)
            return x, v, config.n_discrete_steps

    return sscs_sampler
