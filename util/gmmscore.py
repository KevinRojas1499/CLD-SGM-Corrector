import numpy as np
import yaml
import torch
from math import  pi

def get_score(config, sde):

    params = yaml.safe_load(open('./configs/gmm_config.yaml'))
    c =torch.tensor(params['coeffs']).to('cuda')
    means = torch.tensor(params['means']).to('cuda')
    var = torch.tensor(params['variances']).to('cuda')
    n = len(c)
    var_v = torch.tensor([1.],device='cuda')

    def get_relevant(x, t, i):
        m, v = sde.mean_and_var(means[i],t, var0x=var[i], var0v=var_v)
        vxx = v[0][0]
        vxv = v[1][0]
        vvv = v[2][0]
        vaaar = torch.tensor([[vxx, vxv],[vxv,vvv]])
        inv = torch.linalg.inv(vaaar).to('cuda')
        shift = (inv @ (x-m).T).T
        return vxx*vvv-vxv**2,shift
    
    def density(x,t):
        res = 0
        for i in range(n):
            det, shift = get_relevant(x, t, i)
            res += c[i] * torch.exp(-shift**2/2)/((2. * pi)**2 * det)**.5
        return res

    def score(x,t):
        res = 0
        dens = density(x,t)
        for i in range(n):
            det, shift = get_relevant(x, t, i)
            res += c[i] * torch.exp(-shift**2/2)/((2. * pi)**2 * det)**.5 * -shift
        return res/dens

    return score


