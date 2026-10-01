"""Strong stochastic flow maps for dX=-X dt + sigma dW.

The first two shifted-Legendre Brownian integrals have variances h and h/3.
Chen composition makes coarse and fine evaluations use the same path.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from common import ema_copy, mlp, optimize, time_like


def sample_coefficients(h, count=None):
    h = torch.as_tensor(h)
    if h.ndim == 0:
        if count is None: raise ValueError('count is required for scalar h')
        h = h.expand(count,1)
    scale = torch.cat([h.sqrt(),(h/3).sqrt()],-1)
    return torch.randn_like(scale)*scale


def chen_two(left,right,h_left,h_right):
    h=h_left+h_right
    return torch.stack([left[...,0]+right[...,0],
        (h_left*left[...,1]+h_right*right[...,1]-h_right*left[...,0]+h_left*right[...,0])/h],-1)


class StrongMap(nn.Module):
    def __init__(self,width=64,sigma=.7):
        super().__init__()
        self.net=mlp(5,1,width)
        self.sigma=sigma

    def average_drift(self,x,s,t,coefficients):
        return self.net(torch.cat([x,time_like(s,x),time_like(t,x),coefficients],-1))

    def forward(self,x,s,t,coefficients):
        s,t=time_like(s,x),time_like(t,x)
        return x+(t-s)*self.average_drift(x,s,t,coefficients)+self.sigma*coefficients[:,:1]


def train_stochastic(args,device):
    model=StrongMap(args.width).to(device)
    teacher=ema_copy(model)
    def objective(step):
        B=args.batch_size
        s=.9*torch.rand(B,1,device=device)
        # Exact one-time OU marginals with X0~N(0,1), not simulated trajectories.
        variance=torch.exp(-2*s)+model.sigma**2/2*(1-torch.exp(-2*s))
        x=variance.sqrt()*torch.randn(B,1,device=device)
        h=.001+.019*torch.rand_like(s)
        c=sample_coefficients(h)
        target=x-h*x+model.sigma*c[:,:1]
        matching=((model(x,s,s+h,c)-target).square()/h).mean()
        diagonal=F.mse_loss(model.average_drift(x,s,s,torch.zeros_like(c)),-x)
        t=s+.025+(1-s-.025)*torch.rand_like(s)
        u=(s+t)/2
        L,R=sample_coefficients(u-s),sample_coefficients(t-u)
        C=chen_two(L,R,(u-s).squeeze(-1),(t-u).squeeze(-1))
        if args.ssfm_target=='paper':
            # Paper Algorithm 1: EMA direct target; differentiate both splits.
            with torch.no_grad():direct=teacher(x,s,t,C)
            split=model(model(x,s,u,L),u,t,R)
        else:
            # Released code: EMA split target; differentiate the direct map.
            with torch.no_grad():split=teacher(teacher(x,s,u,L),u,t,R)
            direct=model(x,s,t,C)
        consistency=((split-direct).square()/(t-s)).mean()
        ramp=min(1.,(step+1)/max(1,args.train_steps//4))
        return diagonal+matching+ramp*consistency,{'drift':diagonal,'small_step':matching,'strong_consistency':consistency}
    logs=optimize(model,objective,args.train_steps,args.lr,teacher)
    return model,{'model':model.state_dict(),'sigma':model.sigma},logs


def aggregate_tree(coefficients):
    """Coefficients are [B,leaves,2], with uniform leaf lengths."""
    c=coefficients
    h=1/c.shape[1]
    while c.shape[1]>1:
        c=chen_two(c[:,::2],c[:,1::2],h,h)
        h*=2
    return c[:,0]


@torch.no_grad()
def evaluate_stochastic(model,count,steps,device):
    if steps & (steps-1):raise ValueError('SSFM sample steps must be a power of two.')
    leaves=max(512,steps)
    c=sample_coefficients(torch.tensor(1/leaves,device=device),count*leaves).reshape(count,leaves,2)
    x0=torch.randn(count,1,device=device)
    reference=x0.clone()
    for i in range(leaves):reference=reference-reference/leaves+model.sigma*c[:,i,:1]
    all_coeff=aggregate_tree(c)
    direct=model(x0,0.,1.,all_coeff)
    x=x0.clone()
    block=leaves//steps
    for i in range(steps):
        # aggregate_tree assumes total duration one; equal-half Chen weights are
        # scale invariant, so its output is valid for these smaller blocks too.
        ci=aggregate_tree(c[:,i*block:(i+1)*block])
        x=model(x,i/steps,(i+1)/steps,ci)
    exact_variance=math.exp(-2)+model.sigma**2/2*(1-math.exp(-2))
    return x,{'same_noise_reference_rmse':float((x-reference).square().mean().sqrt()),
              'direct_vs_split_rmse':float((direct-x).square().mean().sqrt()),
              'one_step_reference_rmse':float((direct-reference).square().mean().sqrt()),
              'reference_em_steps':leaves,'sample_variance':float(x.var()),
              'exact_terminal_variance':exact_variance,
              'reference_note':'Same Brownian increments; EM reference retains finite discretization error.'}
