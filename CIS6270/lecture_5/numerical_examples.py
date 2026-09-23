"""Transparent DNA path, rate, geometry, and guidance calculations."""
import math
import torch
from lecture_core import encode, gat_rates, fisher_path, dirichlet_velocity

print('Alphabet order: A C G T.')
source=encode(['TGCA']);target=encode(['ACGT']);u=torch.tensor([[.1,.7,.2,.9]])
print('Gat sampled t=.25 intermediate:',torch.where(u<.25,target,source).tolist(),'= AGGA')
print('Gat target probability loss:',-math.log(.7*.6*.8*.5))
rates=gat_rates(torch.tensor([[[.1,.6,.2,.1]]]),torch.tensor([[2]]),torch.tensor([.5]))
p=.1*rates;p[...,2]=1-.1*rates.sum(-1)
print('Gat Euler probabilities from G:',p.flatten().tolist())
z=torch.tensor([[[.1,.2,.4,.3]]]);posterior=torch.tensor([[[0.,0.,1.,0.]]])
v=dirichlet_velocity(z,posterior,3.)
print('Dirichlet conditional G velocity:',v.flatten().tolist())
y,v=fisher_path(torch.full((1,1,4),.25),torch.tensor([[2]]),torch.tensor([.5]))
print('Fisher midpoint probabilities:',y.square().flatten().tolist())
print('Fisher spherical velocity:',v.flatten().tolist())
a=torch.tensor([0.,math.log(2),math.log(4),0.]);z=a.softmax(-1)
print('Gumbel realized softmax at tau=1:',z.tolist())
print('Same noisy logits at tau=.5:',(2*a).softmax(-1).tolist())
print('Temperature derivative -1 velocity:',(z*(a-(z*a).sum())).tolist())
for beta in [1.,4.,8.]:
 print('DNA endpoint winner probability, beta=',beta,':',math.exp(beta)/(math.exp(beta)+3))
print('ReDi independent matching-pair TC:',math.log(4),'nats; deterministic informative coupling TC: 0')
print('Preference (.8,.2), change (.3,-.1): weighted gain .22 despite second-objective loss')
