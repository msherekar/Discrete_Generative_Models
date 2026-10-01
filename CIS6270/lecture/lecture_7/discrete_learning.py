"""Learn the discrete Markov projections used inside DDSBM and CSBM.
Run: python discrete_learning.py --output results
The outer exact IMF loop is in ot_sbm_examples.py; this file tests learned
rate/transition fitting at its known small bridge solution.
"""
import argparse,json,math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from ot_sbm_examples import ctmc_bridge_example,finite_bridge_example
from learned_bridges import save_artifact

torch.set_num_threads(1)
def kernel(dt):
    z=torch.exp(-3*dt)
    return torch.stack([1/3+2*z/3,2*(1-z)/3,(1-z)/3,2/3+z/3],-1).reshape(-1,2,2)

def rate_model():
    return nn.Sequential(nn.Linear(3,32),nn.SiLU(),nn.Linear(32,32),nn.SiLU(),nn.Linear(32,1),nn.Softplus())

def fit_ddsbm(seed=13,updates=1400,batch=512,checkpoint=None):
    torch.manual_seed(seed);exact=ctmc_bridge_example();pi=torch.tensor(exact['coupling'].reshape(-1),dtype=torch.float32)
    K=kernel(torch.tensor([1.]))[0];Goff=torch.tensor([2.,1.]);model=rate_model();opt=torch.optim.Adam(model.parameters(),lr=.002)
    for _ in range(updates):
        pair=torch.multinomial(pi,batch,replacement=True);a=pair//2;z=pair%2;t=.02+.96*torch.rand(batch)
        left=kernel(t);right=kernel(1-t);idx=torch.arange(batch)
        # P(X_t=x | X_0=a, X_1=z) under the reference bridge.
        bridge=left[idx,a,:]*right[idx,:,z]/K[a,z,None]
        x=torch.multinomial(bridge,1).squeeze(1);y=1-x
        target=Goff[x]*right[idx,y,z]/right[idx,x,z]
        q=model(torch.cat([t[:,None],F.one_hot(x,2).float()],1)).squeeze(1)
        loss=(q-target*q.clamp_min(1e-8).log()).mean()
        opt.zero_grad();loss.backward();opt.step()
    # Compare predicted rates with the independently computed Doob transform.
    truth=[];pred=[]
    with torch.no_grad():
        for row in exact['values']:
            t=float(row['t'])
            for x in [0,1]:
                pred.append(float(model(torch.tensor([[t,float(x==0),float(x==1)]]))))
                truth.append(float(row['rates'][x,1-x]))
    rel=np.mean(((np.array(pred)-truth)/truth)**2)
    save_artifact(checkpoint,method='ddsbm',model=model.state_dict(),steps=100)
    return dict(predicted_off_diagonal_rates=pred,exact_off_diagonal_rates=truth,relative_mse=float(rel),
                approximation='Neural Markov rate projection for a known finite-state SB coupling; outer IMF is separately enumerated exactly.')

def fit_csbm(seed=14,updates=600,checkpoint=None):
    torch.manual_seed(seed);ex=finite_bridge_example(steps=2);paths=torch.tensor(ex['paths']);mass=torch.tensor(ex['bridge'],dtype=torch.float32)
    logits=nn.Parameter(torch.zeros(2,2,2));opt=torch.optim.Adam([logits],lr=.04)
    for _ in range(updates):
        loss=torch.tensor(0.)
        for k in range(2):
            logq=logits[k].log_softmax(-1)
            loss-=torch.sum(mass*logq[paths[:,k],paths[:,k+1]])
        opt.zero_grad();loss.backward();opt.step()
    q=logits.softmax(-1).detach().numpy();truth=np.array(ex['Q']);p=np.array([.5,.5])
    for transition in q:p=p@transition
    save_artifact(checkpoint,method='csbm',transitions=torch.tensor(q),steps=2)
    return dict(transitions=q.tolist(),maximum_transition_error=float(abs(q-truth).max()),terminal=p.tolist(),
                approximation='Exact path-weighted categorical cross-entropy with a fully expressive transition table; no sampling error in this fit.')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='results');args=parser.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    result={'ddsbm':fit_ddsbm(),'csbm':fit_csbm()};(out/'discrete_learned_results.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
