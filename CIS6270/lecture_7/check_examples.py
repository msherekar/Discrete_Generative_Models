"""Independent numerical checks of the lecture's identities and saved runs."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from scipy.linalg import expm
from ot_sbm_examples import *
from learned_bridges import sf2m_targets

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--results',default='results');args=parser.parse_args();out=Path(args.results);out.mkdir(parents=True,exist_ok=True);checks={}
    def check(name,value):
        checks[name]=bool(value)
        if not value:raise AssertionError(name)
    ot=transport_example();check('independent LP and dual certificate',abs(ot['ot_cost']-ot['dual'])<1e-10 and abs(ot['dual']-3.4)<1e-10)
    check('Sinkhorn endpoint constraints',np.allclose(ot['entropic_plan'].sum(1),ot['a']) and np.allclose(ot['entropic_plan'].sum(0),ot['b']))
    br=finite_bridge_example();check('Doob transition normalization',all(np.allclose(q.sum(1),1) for q in br['Q']))
    check('bridge endpoint and path normalization',np.allclose(br['marginals'][-1],[.2,.8]) and abs(br['bridge'].sum()-1)<1e-10)
    # Conditional path laws are identical when the two endpoints are fixed.
    for a in [0,1]:
        for b in [0,1]:
            ix=(br['paths'][:,0]==a)&(br['paths'][:,-1]==b)
            check(f'reference conditional bridges {a}{b}',np.allclose(br['bridge'][ix]/br['bridge'][ix].sum(),br['reference'][ix]/br['reference'][ix].sum()))
    imf=discrete_imf_example();check('IMF converges to full path law',imf['final_path_l1']<1e-10)
    check('IMF path KL is nonincreasing',np.max(np.diff(imf['kl_to_bridge']))<1e-12)
    # Independently differentiate analytic marginals to check the master equation.
    G=np.array([[-2.,2.],[1.,-1.]]);a=np.array([.5,.5]);b=np.array([.2,.8]);pi,f,g,_=sinkhorn(a,b,np.log(a[:,None]*expm(G)))
    p=lambda t:((a*f)@expm(t*G))*(expm((1-t)*G)@g)
    h=expm(.5*G)@g;Q=G*h[None,:]/h[:,None];np.fill_diagonal(Q,0);np.fill_diagonal(Q,-Q.sum(1))
    check('CTMC rates satisfy the independent master equation',np.allclose((p(.500001)-p(.499999))/.000002,p(.5)@Q,atol=1e-8))
    t=torch.tensor([[.1],[.3],[.8]],dtype=torch.float64);x0=torch.tensor([[0.],[1.],[-1.]],dtype=torch.float64);x1=x0+2;z=torch.tensor([[.7],[-.4],[1.2]],dtype=torch.float64)
    x,v,s,sd=sf2m_targets(t,x0,x1,z)
    check('conditional velocity plus score equals bridge drift',torch.allclose(v+.5*s,(x1-x)/(1-t)))
    rw=reward_bridge_example();check('exact path importance identity',np.allclose(rw['weighted_paths'],rw['target_paths']))
    bg=branching_example();check('analytic branch allocation is positive and conserved',np.min(bg['weights'])>=0 and np.allclose(bg['total_mass'],1))
    geo=entangled_geometry_example();check('cone projection and finite-step bound',abs(geo['orthogonal']@geo['unit_direction'])<1e-12 and geo['distance2_after']<geo['distance2_before'])
    d=geo['direction'];v=geo['bias'];dt=geo['max_dt']*1.1
    check('oversized inward step can increase distance',(d-dt*v)@(d-dt*v)>d@d)
    for key in ['dsb','dsbm','sf2m','tr2d2','branch','entangled']:
        path=out/(key+'_learned.json')
        if not path.exists():continue
        r=json.load(open(path))
        if key=='dsb':q=r['history'][-1];check('learned DSB Gaussian endpoint',abs(q['terminal_mean']-2)<.1 and abs(q['terminal_variance']-1)<.15)
        if key=='dsbm':check('learned DSBM Gaussian endpoint',abs(r['terminal_mean']-2)<.1 and abs(r['terminal_variance']-1)<.15)
        if key=='sf2m':check('learned SF2M fields and endpoint',r['velocity_mse']<.05 and r['score_mse']<.05 and abs(r['terminal_variance']-1)<.15)
        if key=='tr2d2':check('full-support search WDCE improves reference reward',r['history'][-1]['reward']>r['reference_reward'] and r['history'][-1]['target_l1']<.5)
        if key=='branch':check('soft branch constraints have small residuals',r['max_mass_error']<.02 and r['minimum_weight']>-.002)
        if key=='entangled':check('learned cone bias retains local alignment',r['min_bias_alignment']>=-1e-6)
    path=out/'discrete_learned_results.json'
    if path.exists():
        r=json.load(open(path));check('learned DDSBM rate projection',r['ddsbm']['relative_mse']<.08);check('learned CSBM transition projection',r['csbm']['maximum_transition_error']<.001)
    (out/'checks.json').write_text(json.dumps(checks,indent=2));print(len(checks),'checks passed')
if __name__=='__main__':main()
