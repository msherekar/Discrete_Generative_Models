"""CIS 6270 Lecture 7. Small, executable OT and bridge calculations.

Run: python ot_sbm_examples.py --output results
The examples use synthetic distributions. Paper-scale benchmarks remain in
original repositories. Every displayed numerical result is recomputed here.
"""
from pathlib import Path
import argparse,itertools,json
import numpy as np
from scipy.special import logsumexp
from scipy.optimize import linprog
from scipy.linalg import expm


def sinkhorn(a,b,log_kernel,iterations=2000,tol=1e-12):
    """Scale a positive kernel to prescribed marginals in log space."""
    a,b=np.asarray(a,float),np.asarray(b,float)
    if np.any(a<=0) or np.any(b<=0):raise ValueError('Use positive marginals on the retained support.')
    if not np.isclose(a.sum(),b.sum()):raise ValueError('Marginals must have equal mass.')
    log_v=np.zeros_like(b);history=[]
    for k in range(iterations):
        log_u=np.log(a)-logsumexp(log_kernel+log_v[None,:],axis=1)
        log_v=np.log(b)-logsumexp(log_kernel+log_u[:,None],axis=0)
        pi=np.exp(log_u[:,None]+log_kernel+log_v[None,:])
        error=max(abs(pi.sum(1)-a).max(),abs(pi.sum(0)-b).max())
        history.append(float(error))
        if error<tol:break
    return pi,np.exp(log_u),np.exp(log_v),history


def transport_example():
    a=np.array([.6,.4]);b=np.array([.3,.7]);C=np.array([[1.,9.],[1.,1.]])
    A=np.array([[1,1,0,0],[0,0,1,1],[1,0,1,0],[0,1,0,1]])
    fit=linprog(C.ravel(),A_eq=A,b_eq=np.r_[a,b],bounds=(0,None),method='highs')
    assert fit.success
    pi,u,v,history=sinkhorn(a,b,-C/2)
    K=np.exp(-C/2);u1=a/K.sum(1);v1=b/(K.T@u1)
    f=np.array([0.,-8.]);g=np.array([1.,9.])
    dual=float(a@f+b@g)
    return dict(a=a,b=b,cost=C,plan=fit.x.reshape(2,2),ot_cost=fit.fun,dual=dual,
                entropic_plan=pi,entropic_cost=float((pi*C).sum()),kernel=K,
                first_u=u1,first_v=v1,first_plan=u1[:,None]*K*v1[None,:],
                marginal_errors=history)


def finite_bridge_example(steps=2):
    a=np.array([.5,.5]);b=np.array([.2,.8]);R=np.array([[.8,.2],[.3,.7]])
    K=np.linalg.matrix_power(R,steps)
    pi,u,v,_=sinkhorn(a,b,np.log(a[:,None]*K))
    h=[np.linalg.matrix_power(R,steps-k)@v for k in range(steps+1)]
    Q=[R*h[k+1][None,:]/h[k][:,None] for k in range(steps)]
    p=[a]
    for q in Q:p.append(p[-1]@q)
    paths=np.array(list(itertools.product(range(2),repeat=steps+1)))
    ref=np.array([a[x[0]]*np.prod([R[x[k],x[k+1]] for k in range(steps)]) for x in paths])
    star=np.array([ref[i]*u[x[0]]*v[x[-1]] for i,x in enumerate(paths)])
    kl=float(np.sum(star*np.log(star/ref)))
    return dict(R=R,K=K,pi=pi,u=u,v=v,h=h,Q=Q,marginals=p,paths=paths,reference=ref,bridge=star,kl=kl)


def discrete_imf_example(steps=3,iterations=80):
    """Exact Markovian and reciprocal projections on a 16-path space."""
    a=np.array([.5,.5]);b=np.array([.2,.8]);R=np.array([[.8,.2],[.3,.7]])
    paths=np.array(list(itertools.product(range(2),repeat=steps+1)))
    ref=np.array([a[x[0]]*np.prod([R[x[k],x[k+1]] for k in range(steps)]) for x in paths])
    joint=a[:,None]*np.linalg.matrix_power(R,steps)
    pi,_,_,_=sinkhorn(a,b,np.log(joint))
    star=np.array([ref[i]*pi[x[0],x[-1]]/joint[x[0],x[-1]] for i,x in enumerate(paths)])
    p=np.array([ref[i]*a[x[0]]*b[x[-1]]/joint[x[0],x[-1]] for i,x in enumerate(paths)])
    hist=[]
    for it in range(iterations):
        qs=[]
        for k in range(steps):
            J=np.zeros((2,2))
            for mass,x in zip(p,paths):J[x[k],x[k+1]]+=mass
            qs.append(J/J.sum(1,keepdims=True))
        m=np.array([a[x[0]]*np.prod([qs[k][x[k],x[k+1]] for k in range(steps)]) for x in paths])
        end=np.zeros((2,2))
        for mass,x in zip(m,paths):end[x[0],x[-1]]+=mass
        p=np.array([ref[i]*end[x[0],x[-1]]/joint[x[0],x[-1]] for i,x in enumerate(paths)])
        hist.append(float(np.sum(p*np.log(p/star))))
    return dict(kl_to_bridge=hist,final_path_l1=float(abs(p-star).sum()),paths=paths,bridge=star,final=p)


def ctmc_bridge_example():
    """Doob rates and exact marginal verification with matrix exponentials."""
    a=np.array([.5,.5]);b=np.array([.2,.8]);G=np.array([[-2.,2.],[1.,-1.]])
    K=expm(G);pi,u,v,_=sinkhorn(a,b,np.log(a[:,None]*K))
    values=[]
    for t in [0,.25,.5,.75,1]:
        h=expm((1-t)*G)@v
        forward=(a*u)@expm(t*G)
        p=forward*h
        rate=G*h[None,:]/h[:,None]
        np.fill_diagonal(rate,0);np.fill_diagonal(rate,-rate.sum(1))
        values.append(dict(t=t,h=h,p=p,rates=rate))
    return dict(generator=G,transition=K,coupling=pi,values=values)


def gaussian_example(epsilon=1.):
    """Brownian reference dX=sqrt(epsilon)dW, N(0,1) to N(2,1)."""
    covariance=(np.sqrt(epsilon**2+4)-epsilon)/2
    t=np.linspace(0,1,101)
    variance=(1-t)**2+t*t+2*t*(1-t)*covariance+epsilon*t*(1-t)
    # Forward Markov drift beta=(y-x)/(1-t) averaged conditionally on X_t.
    cov_yx=(1-t)*covariance+t
    slope=(cov_yx/variance-1)/np.maximum(1-t,1e-10)
    slope[-1]=1-covariance-epsilon
    intercept=2-slope*(2*t)
    return dict(t=t,mean=2*t,variance=variance,covariance=covariance,slope=slope,intercept=intercept)


def reward_bridge_example():
    """A fully masked root has one initial state, so terminal tilting preserves it."""
    base=np.array([.5,.3,.2]);reward=np.log(np.array([1.,2.,4.]));alpha=1.
    target=base*np.exp(reward/alpha);target/=target.sum()
    ref_path=np.array([.3,.2,.18,.12,.12,.08])
    terminal=np.array([0,0,1,1,2,2])
    tilted=ref_path*np.exp(reward[terminal]);tilted/=tilted.sum()
    proposal=np.array([.1,.1,.15,.15,.2,.3])
    log_weight=reward[terminal]+np.log(ref_path)-np.log(proposal)
    exact=np.sum(proposal*np.exp(log_weight))
    weighted=proposal*np.exp(log_weight)/exact
    return dict(base=base,reward=reward,target=target,reference_paths=ref_path,
                target_paths=tilted,proposal=proposal,log_weight=log_weight,weighted_paths=weighted,
                kl=float(np.sum(target*np.log(target/base))))


def branching_example():
    """Exact illustrative mass transfer; neural four-stage version is separate."""
    t=np.linspace(0,1,101);w=np.stack([1-t,.6*t,.4*t],1)
    growth=np.tile([-1.,.6,.4],(len(t),1))
    x=np.stack([t,.7*t+1.2*t*t,.7*t-1.2*t*t],1)
    return dict(t=t,weights=w,growth=growth,positions=x,total_mass=w.sum(1),midpoint_weights=w[50])


def entangled_geometry_example():
    """Check the cone guarantee for the bias increment, separately from noise."""
    s=np.array([3.,4.]);shat=s/np.linalg.norm(s);h=np.array([2.,-1.]);alpha=.8
    orthogonal=h-shat*(shat@h);bias=alpha*shat+orthogonal
    max_dt=2*(s@bias)/(bias@bias)
    dt=.1;before=s@s;after=(s-dt*bias)@(s-dt*bias)
    return dict(direction=s,unit_direction=shat,orthogonal=orthogonal,bias=bias,
                alignment=float(s@bias),max_dt=float(max_dt),dt=dt,distance2_before=float(before),distance2_after=float(after))


def serial(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {k:serial(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [serial(v) for v in x]
    return x


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='results');args=parser.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    results={k:f() for k,f in [('transport',transport_example),('finite_bridge',finite_bridge_example),
      ('discrete_imf',discrete_imf_example),('ctmc',ctmc_bridge_example),('gaussian',gaussian_example),
      ('reward',reward_bridge_example),('branch',branching_example),('entangled',entangled_geometry_example)]}
    (out/'results.json').write_text(json.dumps(serial(results),indent=2))
    for k,v in results.items():print(k,'complete')
    print('OT cost',results['transport']['ot_cost'],'bridge terminal',results['finite_bridge']['marginals'][-1],
          'IMF path L1',results['discrete_imf']['final_path_l1'])
if __name__=='__main__':main()
