# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License for
# CLD-SGM. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import torch
import torch.nn as nn
from . import utils
from torch.distributions.multivariate_normal import MultivariateNormal


@utils.register_model(name='gmm')
class GMM(nn.Module):
    def __init__(self,
                 config):

        super().__init__()

        n1 = nn.Linear(2,3)

        c = torch.tensor([.2,.2,.2,.2,.2]).to(torch.float64)
        n = c.shape[0]
        m = config.mean
        b = config.intercept
        means_x = torch.tensor([[m-b, m -b],[m-b,m+b],[m + b,m - b],[m+b,m+b],[m,m]]).to(torch.float64)
        variances_x = torch.tensor([[[2,0],[0,2]], [[2,1],[1,2]],[[2,0],[0,2]], [[2,1],[1,2]],[[1,0],[0,1]]]).to(torch.float64)

        means = torch.zeros((n,4)).to(torch.float64)
        variances = torch.zeros((n,4,4)).to(torch.float64)

        for i in range(n):
            means[i,0:2] = means_x[i]
            variances[i,0:2,0:2] = variances_x[i]
        
        means = means.to('cuda')
        variances = variances.to('cuda')

        self.gmm = GaussianMixture(c,means,variances)

    def forward(self, u, t, sde):
        return self.gmm.score(u,t, sde)
    

class GaussianMixture():

    def __init__(self,c, means, variances):
        self.n = c.shape[0]
        self.c = c
        self.gaussians = [Gaussian(means[i],variances[i]) for i in range(self.n)]
    
    def eval(self,x,t, sde):
        density = 0
        for i in range(self.n):
            density+= self.c[i]*self.gaussians[i].eval(x,t,sde)
        return density
    
    def score(self,x,t, sde):
        score = 0
        for i in range(self.n):
            score+= self.c[i]*self.gaussians[i].score_contribution(x,t,sde)
        return score/self.eval(x,t, sde).unsqueeze(-1)


class Gaussian():

    def batch_mv(self,bmat, bvec):
        return torch.matmul(bmat, bvec.unsqueeze(-1)).squeeze(-1)

    def __init__(self,mean,variance):
        self.d = mean.shape[0]
        self.mean = mean 
        self.var = variance   
        self.varx = self.var[0:2,0:2]
        self.varv = self.var[2:4,2:4]

    def eval(self,x,t,sde, params=None):
        mean_t, var_t = self.parameters_for_eval(t,sde) if params == None else params
        normal = MultivariateNormal(mean_t,covariance_matrix=var_t)
        
        try:
            density = torch.exp(normal.log_prob(x))
        except:
            print("VALUES OUT OF DISTRIBUTION")
            density = torch.zeros(x.shape[0], device='cuda')
        return density
    
    def score_contribution(self,x,t, sde):
        mean_t, var_t = self.parameters_for_eval(t,sde)
        shift = x - mean_t
        return - self.eval(x,t, sde,params=(mean_t,var_t)).unsqueeze(-1) * self.batch_mv(torch.linalg.inv(var_t),shift)

    def parameters_for_eval(self, t, sde):
        t =  torch.tensor([t[0]]).to('cuda')
        mean_t, v = sde.mean_and_var(self.mean.unsqueeze(0),t, var0x=self.varx, var0v=self.varv)
        n = v[0].shape[0]
        vv = torch.zeros((2*n,2*n),dtype=torch.float64).to('cuda')
        vv[0:2, 0:2] = v[0]
        vv[2:4,0:2] = v[1]
        vv[0:2,2:4] = v[1]
        vv[2:4,2:4] = v[2]
        return mean_t,vv