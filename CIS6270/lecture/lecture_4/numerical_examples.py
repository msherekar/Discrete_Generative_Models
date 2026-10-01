"""Print the four-letter training and sampling calculations from the lecture."""
import math
import torch
from lecture_core import udlm_rates, geometric_cfg

dtype = torch.float64
print('Alphabet order: A C G T; log losses are nats.')
q = .6*torch.eye(4,dtype=dtype)+.1*torch.ones(4,4,dtype=dtype)
print('Uniform one-step corruption matrix:\n',q)
print('Two-step marginal from clean A:',(q@q)[0].tolist())
reverse=q[0]*q[:,1];reverse/=reverse.sum()
print('Previous base given clean A and final C:',reverse.tolist())
loss=2*(-math.log(.6)-math.log(.7))
print('MDLM ACGT -> AmGm, t=.5, loss:',loss)
print('MDLM reverse .5 -> .25 at missing C:',[.05,.30,.10,.05,.50])
z=torch.tensor([[1]]);t=torch.tensor([.5],dtype=dtype)
a=udlm_rates(torch.tensor([[[1,0,0,0]]],dtype=dtype),z,t)
b=udlm_rates(torch.tensor([[[.6,.2,.1,.1]]],dtype=dtype),z,t)
rate_kl=(a*(a.clamp_min(1e-12).log()-b.clamp_min(1e-12).log())+b-a).sum()
print('UDLM target rates C -> A,C,G,T:',a.flatten().tolist())
print('UDLM learned rates:',b.flatten().tolist(),'rate loss:',float(rate_kl))
p=.1*b;p[...,1]=1-.1*b.sum(-1)
print('UDLM Euler step:',p.flatten().tolist())
print('CFG strength 2:',geometric_cfg(torch.full((4,),.25),torch.tensor([.1,.2,.6,.1]),2).tolist())
