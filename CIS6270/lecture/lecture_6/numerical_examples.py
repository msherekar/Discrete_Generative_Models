"""CIS 6270 Lecture 6. Small, reproducible flow-map teaching experiments.

Run: python numerical_examples.py --output outputs/numerical
Dependencies: Python 3.11+, numpy, torch.
The toy experiments validate the lecture mathematics. They do not reproduce
the large-scale training or benchmark claims of the cited papers.
"""
import argparse, json, math, pathlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

def flow(model, x, s, t):
    """Residual map with an exactly identity diagonal."""
    return x + (t-s)*model(torch.cat([x,s,t],dim=-1))

def train_scalar(output, iterations=6000):
    """Self-distill the flow of dx/dt=x, using diagonal and composition only."""
    torch.manual_seed(6270)
    torch.set_num_threads(2)
    model=nn.Sequential(nn.Linear(3,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU(),nn.Linear(64,1))
    opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    history=[]
    for step in range(iterations):
        # Broad state coverage includes states reached by the split map.
        x=6*torch.rand(256,1)-3
        times=torch.rand(256,2).sort(dim=-1).values
        s,t=times[:,:1],times[:,1:]
        u=s+(t-s)*torch.rand_like(s)
        diagonal=model(torch.cat([x,s,s],dim=-1))
        diag_loss=(diagonal-x).square().mean()
        with torch.no_grad():
            target=flow(model,flow(model,x,s,u),u,t)
        prediction=flow(model,x,s,t)
        # Normalize the interval residual to prevent tiny intervals dominating
        # the count of nearly zero-error examples. Keep a finite floor.
        cons_loss=((prediction-target)/(t-s).clamp_min(0.1)).square().mean()
        loss=diag_loss + (0 if step<500 else 1)*cons_loss
        opt.zero_grad();loss.backward();opt.step()
        if step%500==0:history.append({'step':step,'diagonal':diag_loss.item(),'composition':cons_loss.item()})
    with torch.no_grad():
        x=torch.linspace(-1.5,1.5,501)[:,None]
        s=torch.zeros_like(x);t=torch.ones_like(x);u=0.5*t
        pred=flow(model,x,s,t);truth=x*math.e
        split=flow(model,flow(model,x,s,u),u,t)
        rmse=(pred-truth).square().mean().sqrt().item()
        composition_rmse=(pred-split).square().mean().sqrt().item()
        np.savez(output/'scalar_predictions.npz',x=x.numpy().ravel(),pred=pred.numpy().ravel(),truth=truth.numpy().ravel())
    torch.save(model.state_dict(),output/'scalar_map_weights.pt')
    return {'endpoint_rmse':rmse,'composition_rmse':composition_rmse,'iterations':iterations,'history':history}

def categorical_map(net,x,s,t):
    psi=net(x,s,t).softmax(dim=-1)
    h=(t-s)/(1-s)
    return (1-h)*x+h*psi,psi

def check_categorical():
    x=torch.tensor([-.2,.6,1.1],dtype=torch.float64)
    psi=torch.tensor([.1,.7,.2],dtype=torch.float64)
    y=(1/3)*x+(2/3)*psi
    assert torch.allclose(y,torch.tensor([0,2/3,.5],dtype=torch.float64))
    target=(1/3)*torch.tensor([.8,.2])+(2/3)*torch.tensor([.2,.8])
    assert torch.allclose(target,torch.tensor([.4,.6]))
    logits=torch.tensor([.2,-.1],requires_grad=True)
    loss=F.kl_div(logits.log_softmax(-1),target,reduction='sum')
    loss.backward()
    assert torch.allclose(logits.grad,logits.softmax(-1)-target,atol=1e-7)
    return {'mapped_state':y.tolist(),'state_sum':y.sum().item(),'target':target.tolist()}

def check_meanflow_jvp():
    # Exact backward average for dz/dt=z. Stay off diagonal for this check.
    def exact_average(z,r,t):
        h=t-r
        return z*(-torch.expm1(-h))/h
    z=torch.tensor([[1.7]],dtype=torch.float64)
    r=torch.tensor([[.2]],dtype=torch.float64)
    t=torch.tensor([[.8]],dtype=torch.float64)
    value,derivative=torch.func.jvp(exact_average,(z,r,t),(z,torch.zeros_like(r),torch.ones_like(t)))
    target=z-(t-r)*derivative
    assert torch.allclose(value,target,atol=1e-10)
    return {'average':value.item(),'identity_residual':(value-target).abs().item()}

def posterior_value_demo():
    # Prior Z~N(0,1), observation x=beta*Z+alpha*eps.
    # Reward r(Z)=c*Z gives an analytic log moment-generating function.
    torch.manual_seed(13)
    alpha,beta,c=.7,.6,.4
    x=torch.tensor(.3,dtype=torch.float64,requires_grad=True)
    gain=beta/(alpha**2+beta**2)
    variance=alpha**2/(alpha**2+beta**2)
    eps=torch.randn(100000,dtype=torch.float64)
    z=gain*x+math.sqrt(variance)*eps
    logw=c*z
    estimate=torch.logsumexp(logw,0)-math.log(len(eps))
    gradient=torch.autograd.grad(estimate,x)[0]
    exact=c*gain*x.detach()+.5*c*c*variance
    exact_gradient=c*gain
    assert abs(gradient.item()-exact_gradient)<1e-10
    assert abs(estimate.item()-exact.item())<.006
    return {'estimated_value':estimate.item(),'exact_value':exact.item(),'gradient':gradient.item(),'exact_gradient':exact_gradient}

def sample_gap_counts(means,remaining_budget):
    """Paper-style bounded proposals followed by left-to-right budget capping.

    These proposals match the per-gap means before joint truncation. They are
    not asserted to identify the full conditional count law from means alone.
    """
    counts=[];remaining=int(remaining_budget)
    for mean in means:
        if remaining_budget==0:count=0
        else:
            prob=float(torch.as_tensor(mean).clamp(0,remaining_budget))/remaining_budget
            count=int(torch.distributions.Binomial(remaining_budget,probs=prob).sample())
        count=min(count,remaining);counts.append(count);remaining-=count
    return torch.tensor(counts,dtype=torch.long)

def insert_by_gap(state,old_birth_times,counts,new_noise,birth_time):
    """Insert ordered noise rows into the n+1 gaps and preserve clock alignment."""
    assert len(counts)==len(state)+1
    pieces=[];clocks=[];offset=0
    for gap,count in enumerate(counts.tolist()):
        if count:
            pieces.append(new_noise[offset:offset+count]);offset+=count
            clocks.append(torch.full((count,),float(birth_time)))
        if gap<len(state):
            pieces.append(state[gap:gap+1]);clocks.append(old_birth_times[gap:gap+1])
    assert offset==len(new_noise)
    return torch.cat(pieces),torch.cat(clocks)

def check_expansion():
    state=torch.tensor([[1.,0.],[0.,1.]])
    births=torch.tensor([0.,.2]);counts=torch.tensor([1,0,1])
    noise=torch.tensor([[-.2,.4],[.3,-.1]])
    expanded,bt=insert_by_gap(state,births,counts,noise,.5)
    local=(.75-bt)/(1-bt)
    assert expanded.shape==(4,2)
    assert torch.allclose(bt,torch.tensor([.5,0.,.2,.5]))
    assert torch.allclose(local,torch.tensor([.5,.75,.6875,.5]))
    return {'birth_times':bt.tolist(),'local_times':local.tolist(),'expected_insertions':((.75-.25)/(1-.25))*3}

def chen_two(left,right,hL,hR):
    h=hL+hR
    return torch.stack([left[...,0]+right[...,0],
        (hL*left[...,1]+hR*right[...,1]-hR*left[...,0]+hL*right[...,0])/h],dim=-1)

def check_brownian():
    left=torch.tensor([.2,.04]);right=torch.tensor([-.1,-.02])
    coarse=chen_two(left,right,.5,.5)
    assert torch.allclose(coarse,torch.tensor([.1,-.14]))
    direct=1+.5+.8*coarse[0]
    split=(1+.25+.8*left[0])+.25+.8*right[0]
    assert torch.allclose(direct,split)
    torch.manual_seed(17)
    scale=torch.tensor([.5,1/6]).sqrt()
    L=torch.randn(200000,2)*scale;R=torch.randn(200000,2)*scale
    C=chen_two(L,R,.5,.5)
    covariance=torch.cov(C.T)
    assert torch.allclose(covariance,torch.diag(torch.tensor([1.,1/3])),atol=.012)
    # Exact polynomial restriction identities, tested over unequal intervals.
    hL,hR=.3,.7
    q=torch.linspace(0,1,100,dtype=torch.float64)
    global_left=2*(hL*q)/(hL+hR)-1
    local_left=(hL/(hL+hR))*(2*q-1)-hR/(hL+hR)
    assert torch.allclose(global_left,local_left,atol=1e-12)
    return {'coarse_coefficients':coarse.tolist(),'same_noise_endpoint':direct.item(),'empirical_covariance':covariance.tolist()}

def check_meta_gradient():
    d=torch.tensor(.3,requires_grad=True)
    w=torch.tensor([1.,4.]);grad_w=torch.tensor([.2,.8]);a=.5
    residual=d+(w-1)*d.detach()-a*grad_w
    residual.square().mean().backward()
    expected=2*(w*d.detach()-a*grad_w).mean()
    assert torch.allclose(d.grad,expected)
    return {'surrogate_gradient':d.grad.item(),'estimating_equation_gradient':expected.item()}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='results');parser.add_argument('--iterations',type=int,default=6000)
    args=parser.parse_args();out=pathlib.Path(args.output);out.mkdir(parents=True,exist_ok=True)
    results={'torch_version':torch.__version__,'categorical':check_categorical(),'meanflow_jvp':check_meanflow_jvp(),
        'posterior_value':posterior_value_demo(),'expansion':check_expansion(),'brownian':check_brownian(),'meta_gradient':check_meta_gradient()}
    results['scalar_training']=train_scalar(out,args.iterations)
    (out/'checks.json').write_text(json.dumps(results,indent=2))
    print(json.dumps({k:v for k,v in results.items() if k!='scalar_training'},indent=2))
    print('Scalar endpoint RMSE',results['scalar_training']['endpoint_rmse'])
    print('Scalar composition RMSE',results['scalar_training']['composition_rmse'])
if __name__=='__main__':main()
