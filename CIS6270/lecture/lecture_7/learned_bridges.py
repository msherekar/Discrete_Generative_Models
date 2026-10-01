"""Small, runnable learning examples for CIS 6270 Lecture 7.

These are synthetic teaching implementations of the stated objectives, not
reproductions of the papers' architectures or benchmark results. Run
  python learned_bridges.py --method all --output results
CPU, no dataset download, fixed seeds. See README.md for each approximation.
"""
import argparse,itertools,json,math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from ot_sbm_examples import serial,gaussian_example,finite_bridge_example

torch.set_num_threads(1)

def mlp(din,dout,width=48):
    return nn.Sequential(nn.Linear(din,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU(),nn.Linear(width,dout))

def affine_fit(x,y):
    slope=np.mean((x-x.mean())*(y-y.mean()))/np.var(x)
    return np.array([slope,y.mean()-slope*x.mean()])

def apply_affine(coeff,x):return coeff[0]*x+coeff[1]

def save_artifact(path, **state):
    """Save only tensors and primitive metadata for weights_only checkpoint loading."""
    if path is not None:
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        torch.save(dict(schema_version=1,**state),path)

def dsb(seed=7,rounds=8,steps=80,batch=16000,checkpoint=None):
    """Algorithm-1 reverse regression, affine maps, noise variance 2 dt."""
    rng=np.random.default_rng(seed);dt=1/steps
    forward=np.tile([1.,0.],(steps,1));backward=forward.copy();history=[]
    def rollout(coeff,mean):
        xs=[rng.normal(mean,1,batch)]
        for c in coeff:xs.append(apply_affine(c,xs[-1])+np.sqrt(2*dt)*rng.normal(size=batch))
        return np.stack(xs,1)
    for epoch in range(rounds):
        xs=rollout(forward,0.)
        for k in range(steps):
            x,y=xs[:,k],xs[:,k+1]
            # Paper Eq. 12: y + F_k(x) - F_k(y).
            target=y+apply_affine(forward[k],x)-apply_affine(forward[k],y)
            backward[steps-1-k]=affine_fit(y,target)
        ys=rollout(backward,2.)
        for j in range(steps):
            y,x=ys[:,j],ys[:,j+1]
            target=x+apply_affine(backward[j],y)-apply_affine(backward[j],x)
            forward[steps-1-j]=affine_fit(x,target)
        z=rollout(forward,0.)
        history.append(dict(iteration=epoch+1,terminal_mean=float(z[:,-1].mean()),terminal_variance=float(z[:,-1].var()),endpoint_covariance=float(np.cov(z[:,0],z[:,-1])[0,1])))
    save_artifact(checkpoint,method='dsb',forward=forward.tolist(),backward=backward.tolist(),steps=steps,epsilon=2.)
    return dict(history=history,reference_epsilon=2.,analytic_endpoint_covariance=math.sqrt(2)-1,
                approximation='Affine regression and finite-step Gaussian reverse kernels; both endpoint resets are executed.')

def dsbm(seed=8,rounds=8,steps=80,batch=16000,checkpoint=None):
    """Alternating Brownian bridge regression, affine fields at grid midpoints.

Forward/reverse endpoint resets control approximation drift. No rank matching
or hidden endpoint correction is applied to the generated samples.
"""
    rng=np.random.default_rng(seed);dt=1/steps;eps=1.
    x0=rng.normal(0,1,batch);x1=rng.normal(2,1,batch);history=[]
    def fit_and_sample(a,b,start_mean,reverse):
        coefs=[]
        for k in range(steps):
            s=(k+.5)/steps;t=1-s if reverse else s
            xt=(1-t)*a+t*b+np.sqrt(eps*t*(1-t))*rng.normal(size=batch)
            target=(a-xt)/t if reverse else (b-xt)/(1-t)
            coefs.append(affine_fit(xt,target))
        start=rng.normal(start_mean,1,batch);x=start.copy()
        for c in coefs:x+=dt*apply_affine(c,x)+np.sqrt(eps*dt)*rng.normal(size=batch)
        return start,x,coefs
    for epoch in range(rounds):
        x0,x1,_=fit_and_sample(x0,x1,0.,False)
        y1,y0,backward=fit_and_sample(x0,x1,2.,True)
        x0,x1=y0,y1
        history.append(dict(iteration=epoch+1,source_mean=float(x0.mean()),source_variance=float(x0.var()),endpoint_covariance=float(np.cov(x0,x1)[0,1])))
    a,b,forward=fit_and_sample(x0,x1,0.,False)
    save_artifact(checkpoint,method='dsbm',forward=np.asarray(forward).tolist(),backward=np.asarray(backward).tolist(),steps=steps,epsilon=eps)
    return dict(history=history,terminal_mean=float(b.mean()),terminal_variance=float(b.var()),analytic_endpoint_covariance=(math.sqrt(5)-1)/2,
                approximation='Gaussian affine drift regression, finite Monte Carlo batches, midpoint coefficients with Euler sampling.')

def sf2m_targets(t,x0,x1,z,epsilon=1.):
    m=(1-t)*x0+t*x1;sigma=(epsilon*t*(1-t)).sqrt();x=m+sigma*z
    v=x1-x0+(1-2*t)/(2*t*(1-t))*(x-m)
    score=-(x-m)/(epsilon*t*(1-t))
    return x,v,score,sigma

def sf2m(seed=9,updates=1800,batch=512,checkpoint=None):
    torch.manual_seed(seed);model=mlp(2,2);opt=torch.optim.Adam(model.parameters(),lr=.002)
    c=(math.sqrt(5)-1)/2;history=[]
    for i in range(updates):
        x0=torch.randn(batch,1);x1=2+c*x0+math.sqrt(1-c*c)*torch.randn_like(x0)
        t=.02+.96*torch.rand_like(x0);x,v,s,sigma=sf2m_targets(t,x0,x1,torch.randn_like(x0))
        vp,sp=model(torch.cat([t,x],1)).chunk(2,1)
        loss=((vp-v).square()+(sigma*(sp-s)).square()).mean()
        opt.zero_grad();loss.backward();opt.step()
        if i%300==0:history.append(float(loss.detach()))
    with torch.no_grad():
        t=.05+.9*torch.rand(5000,1);var=(1-t)**2+t*t+2*t*(1-t)*c+t*(1-t)
        x=2*t+var.sqrt()*torch.randn_like(t);vp,sp=model(torch.cat([t,x],1)).chunk(2,1)
        # Continuity equation of N(2t,var(t)).
        dvar=(-2+4*t)+(2*c+1)*(1-2*t)
        vt=2+.5*dvar/var*(x-2*t);st=-(x-2*t)/var
        errors=dict(velocity_mse=float((vp-vt).square().mean()),score_mse=float((sp-st).square().mean()))
        y=torch.randn(6000,1);dt=1/160
        for k in range(160):
            tt=torch.full_like(y,(k+.5)*dt);v,s=model(torch.cat([tt,y],1)).chunk(2,1)
            y+=(v+.5*s)*dt+math.sqrt(dt)*torch.randn_like(y)
    save_artifact(checkpoint,method='sf2m',model=model.state_dict(),steps=160,epsilon=1.)
    return dict(loss_samples=history,**errors,terminal_mean=float(y.mean()),terminal_variance=float(y.var()),target_mean=2.,target_variance=1.,
                approximation='Small neural velocity/score, exact Gaussian endpoint coupling, finite-step Euler SDE.')

class MaskedModel(nn.Module):
    def __init__(self,length=4,vocab=3):
        super().__init__();self.length=length;self.vocab=vocab
        self.net=mlp(length*(vocab+1),length*vocab)
        with torch.no_grad():
            self.net[-1].weight.zero_();self.net[-1].bias.copy_(torch.tensor([.5,.3,.2]).log().repeat(length))
    def forward(self,x):
        z=F.one_hot(x,self.vocab+1).float().flatten(1)
        return self.net(z).reshape(-1,self.length,self.vocab)

def token_reward(x):
    return .35*(x==2).sum(-1).float()+.25*(x[...,1:]==x[...,:-1]).sum(-1).float()

def tr2d2(seed=10,epochs=5,searches=450,checkpoint=None):
    """Small search + replay + WDCE analogue on abstract tokens.

The search uses a fixed next-position order, a restricted MDM schedule. The
full-support softmax selection probabilities are recorded in the proposal law.
An exact finite reward-tilt calculation lives in ot_sbm_examples.py.
"""
    torch.manual_seed(seed);rng=np.random.default_rng(seed);model=MaskedModel();opt=torch.optim.Adam(model.parameters(),lr=.004);L=4;V=3;alpha=.6
    base=torch.tensor([.5,.3,.2]);allx=torch.tensor(list(itertools.product(range(V),repeat=L)))
    reference=base[allx].prod(-1);target=reference*torch.exp(token_reward(allx)/alpha);target/=target.sum();history=[]
    def prior(prefix):
        x=torch.full((1,L),V,dtype=torch.long);x[0,:len(prefix)]=torch.tensor(prefix,dtype=torch.long)
        with torch.no_grad():return model(x)[0,len(prefix)].softmax(-1).numpy()
    def sample_buffer():
        # Full-support softmax search, k=V. Record the actual selection law.
        # All rollouts are retained so no unrecorded curation changes the proposal.
        nodes={():[0,0.]};completed=[]
        for _ in range(searches):
            prefix=();visited=[()];logq=0.
            while len(prefix)<L:
                p=prior(prefix);children=[prefix+(j,) for j in range(V)]
                n=max(nodes[prefix][0],1)
                values=[]
                for j,ch in enumerate(children):
                    visits,total=nodes.get(ch,[0,0.])
                    values.append(total/max(visits,1)+1.1*p[j]*math.sqrt(n)/(1+visits))
                prob=np.exp(np.array(values)-max(values));prob/=prob.sum()
                j=int(rng.choice(V,p=prob));logq+=math.log(prob[j]);child=children[j]
                new=child not in nodes
                if new:nodes[child]=[0,0.]
                prefix=child;visited.append(prefix)
                if new:break
            while len(prefix)<L:
                p=prior(prefix);j=int(rng.choice(V,p=p));logq+=math.log(max(p[j],1e-12));prefix+=(j,)
            seq=torch.tensor(prefix);reward=float(token_reward(seq))
            for node in visited:nodes[node][0]+=1;nodes[node][1]+=reward
            completed.append((seq,logq))
        return torch.stack([z[0] for z in completed]),torch.tensor([z[1] for z in completed])
    def law():
        p=torch.ones(len(allx))
        with torch.no_grad():
            for k in range(L):
                masked=allx.clone();masked[:,k:]=V
                p*=model(masked)[:,k].softmax(-1).gather(1,allx[:,k,None]).squeeze(1)
        return p/p.sum()
    for epoch in range(epochs):
        seq,logp=sample_buffer();logw=token_reward(seq)/alpha+base[seq].log().sum(-1)-logp
        w=logw.softmax(0).detach()
        for _ in range(100):
            mask=torch.rand(seq.shape)<.5
            empty=~mask.any(dim=1)
            mask[empty,torch.randint(L,(int(empty.sum()),))]=True
            masked=seq.masked_fill(mask,V)
            ce=F.cross_entropy(model(masked).transpose(1,2),seq,reduction='none')
            per=(ce*mask).sum(-1)/mask.sum(-1);loss=(w*per).sum()
            opt.zero_grad();loss.backward();opt.step()
        p=law();history.append(dict(epoch=epoch+1,reward=float(p@token_reward(allx)),base_kl=float((p*(p/reference).log()).sum()),target_l1=float((p-target).abs().sum()),effective_samples=float(1/w.square().sum())))
    save_artifact(checkpoint,method='tr2d2',model=model.state_dict(),length=L,vocab=V,alpha=alpha,steps=L)
    return dict(history=history,reference_reward=float(reference@token_reward(allx)),exact_tilt_reward=float(target@token_reward(allx)),
                approximation='Full-support MCTS with its recorded selection/rollout proposal, finite-batch self-normalized WDCE, fixed reveal order. This replaces the paper-scale curated replay heuristic with a tractable proposal.')

def branch(seed=11,updates=350,checkpoint=None):
    """Four-stage BranchSBM teaching analogue with a fixed endpoint coupling."""
    torch.manual_seed(seed);phi=mlp(6,2);velocity=mlp(6,2);growth=nn.Parameter(torch.tensor([-.8,.45,.3]));target_w=torch.tensor([0.,.6,.4]);w0=torch.tensor([1.,0.,0.])
    def batch(n=192):
        t=torch.rand(n,1,requires_grad=True);k=torch.randint(3,(n,));one=F.one_hot(k,3).float()
        x0=.12*torch.randn(n,2);centers=torch.tensor([[1.,0.],[1.,1.],[1.,-1.]])
        x1=centers[k]+.12*torch.randn(n,2);return t,k,one,x0,x1
    def interpolant(t,one,x0,x1):
        ph=phi(torch.cat([t,one,x0],1));x=(1-t)*x0+t*x1+t*(1-t)*ph
        dx=torch.cat([torch.autograd.grad(x[:,j].sum(),t,create_graph=True,retain_graph=True)[0] for j in range(2)],1)
        return x,dx
    def cost(x):return .3*torch.exp(-((x-torch.tensor([.5,0.]))**2).sum(-1)/.08)
    opt=torch.optim.Adam(phi.parameters(),lr=.002)
    for _ in range(updates):
        t,k,one,x0,x1=batch();x,dx=interpolant(t,one,x0,x1);loss=(.5*dx.square().sum(-1)+cost(x)).mean()
        opt.zero_grad();loss.backward();opt.step()
    opt=torch.optim.Adam(velocity.parameters(),lr=.002)
    for _ in range(updates):
        t,k,one,x0,x1=batch();x,dx=interpolant(t,one,x0,x1)
        pred=velocity(torch.cat([t.detach(),one,x.detach()],1));loss=(pred-dx.detach()).square().mean()
        opt.zero_grad();loss.backward();opt.step()
    opt=torch.optim.Adam([growth],lr=.01)
    def massloss(t):
        w=w0+t*growth;term=(w0+growth-target_w).square().mean()
        return term+((w.sum(-1)-1)**2).mean()+(-w).clamp_min(0).mean()
    for _ in range(updates):
        loss=massloss(torch.rand(100,1));opt.zero_grad();loss.backward();opt.step()
    opt=torch.optim.Adam(list(phi.parameters())+list(velocity.parameters())+[growth],lr=.0003)
    for _ in range(updates):
        t,k,one,x0,x1=batch();x,dx=interpolant(t,one,x0,x1);v=velocity(torch.cat([t,one,x],1))
        w=w0+t*growth;wk=w.gather(1,k[:,None]).squeeze(1)
        energy=(wk*(.5*v.square().sum(-1)+cost(x))).mean()
        loss=energy+20*massloss(t)+10*(v-dx).square().mean()
        opt.zero_grad();loss.backward();opt.step()
    t,k,one,x0,x1=batch(1000);x,dx=interpolant(t,one,x0,x1);v=velocity(torch.cat([t,one,x],1));w=w0+t*growth
    save_artifact(checkpoint,method='branch',interpolant=phi.state_dict(),model=velocity.state_dict(),growth=growth.detach(),steps=100)
    return dict(terminal_weights=(w0+growth).detach().tolist(),target_weights=target_w.tolist(),velocity_fit_mse=float((v-dx).square().mean().detach()),max_mass_error=float((w.sum(-1)-1).abs().max().detach()),minimum_weight=float(w.min().detach()),
                approximation='Small MLP interpolants, fixed synthetic endpoint coupling, linear growth, sampled soft mass constraints.')

class CoupledCone(nn.Module):
    def __init__(self):
        super().__init__();self.net=mlp(5,3,32)
        with torch.no_grad():self.net[-1].weight.mul_(.01);self.net[-1].bias.copy_(torch.tensor([-3.,0.,0.]))
    def forward(self,t,x):
        mean=x.mean(1,keepdim=True).expand_as(x);tt=torch.full_like(x[...,:1],float(t))
        raw=self.net(torch.cat([x,mean,tt],-1));s=1.-x;norm=s.norm(dim=-1,keepdim=True)
        shat=s/norm.clamp_min(1e-8);h=raw[...,1:]
        bias=F.softplus(raw[...,:1])*shat+h-(h*shat).sum(-1,keepdim=True)*shat
        return torch.where(norm>1e-8,bias,torch.zeros_like(bias))

def entangled(seed=12,epochs=60,batch=512,steps=16,checkpoint=None):
    """Interacting normalized particles, terminal potential, Euler path CE."""
    torch.manual_seed(seed);model=CoupledCone();opt=torch.optim.Adam(model.parameters(),lr=.002);dt=1/steps;sigma=.6;history=[]
    def base(x):return -.12*x-.35*(x-x.mean(1,keepdim=True))
    for epoch in range(epochs):
        states=[];nexts=[];lr=[];x=.3*torch.randn(batch,3,2)
        with torch.no_grad():
            for k in range(steps):
                a=base(x);u=model(k*dt,x);mean=x+(a+sigma*u)*dt
                y=mean+sigma*math.sqrt(dt)*torch.randn_like(x)
                r_ref=(y-x-a*dt)/(sigma*math.sqrt(dt));r_pro=(y-mean)/(sigma*math.sqrt(dt))
                lr.append(-.5*(r_ref.square()-r_pro.square()).sum((1,2)));states.append(x);nexts.append(y);x=y
            # A terminal potential, not a separately enforced terminal density.
            logg=-.5*((x-1.)**2).sum((1,2))/1.5
            logw=logg+torch.stack(lr,1).sum(1);w=logw.softmax(0).detach()
        path_nll=torch.zeros(batch)
        for k,(x,y) in enumerate(zip(states,nexts)):
            mean=x+(base(x)+sigma*model(k*dt,x))*dt
            path_nll+=.5*((y-mean)/(sigma*math.sqrt(dt))).square().sum((1,2))
        loss=(w*path_nll).sum();opt.zero_grad();loss.backward();opt.step()
        if epoch%10==0:history.append(dict(epoch=epoch+1,weighted_nll=float(loss.detach()),effective_samples=float(1/w.square().sum()),terminal_mean=float(nexts[-1].mean())))
    x=torch.randn(400,3,2);u=model(.5,x);alignment=(u*(1-x)).sum(-1)
    save_artifact(checkpoint,method='entangled',model=model.state_dict(),steps=steps,sigma=sigma)
    return dict(history=history,min_bias_alignment=float(alignment.min().detach()),
                approximation='Mean-field coupled network, synthetic overdamped Euler chain, self-normalized CE, terminal potential rather than an enforced density.')

METHODS=dict(dsb=dsb,dsbm=dsbm,sf2m=sf2m,tr2d2=tr2d2,branch=branch,entangled=entangled)
def main():
    p=argparse.ArgumentParser();p.add_argument('--method',choices=['all',*METHODS],default='all');p.add_argument('--output',default='results');args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    result={}
    for key,fn in METHODS.items():
        if args.method not in ['all',key]:continue
        result[key]=fn();(out/(key+'_learned.json')).write_text(json.dumps(serial(result[key]),indent=2));print(key,'complete',flush=True)
    (out/'learned_results.json').write_text(json.dumps(serial(result),indent=2))
if __name__=='__main__':main()
