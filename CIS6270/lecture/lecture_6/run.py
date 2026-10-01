#!/usr/bin/env python3
"""Train, save, reload, and sample the Lecture 6 flow-map methods."""
import argparse
import json
import platform
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from common import (MapNet, SequenceNet, interpolate, load_text, mixture_data,
                    mixture_metrics, seed_all, text_metrics, write_json)
from continuous import (Autoencoder, embed_surface, integrate_velocity, sample_continuous,
                        train_continuous, train_latent)
from categorical import sample_categorical, train_categorical
from posterior import (guided_samples, posterior_diagnostics, posterior_samples,
                       posterior_value, reward, train_posterior, train_reward_drift,
                       weighted_diamond_samples)
from expanding import ExpandingNet, sample_expanding, train_expanding
from stochastic import StrongMap, evaluate_stochastic, train_stochastic

ROOT = Path(__file__).resolve().parent
CONTINUOUS = ['flow-matching','fmm-lagrangian','fmm-eulerian','self-distill',
              'consistency','shortcut','meanflow','latent']
CATEGORICAL = ['fmlm','categorical','discrete-lsd','discrete-esd']
METHODS = CONTINUOUS+CATEGORICAL+['diamond','meta','expanding','ssfm']


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method',choices=METHODS)
    p.add_argument('--mode',choices=['train-sample','train','sample'],default='train-sample')
    p.add_argument('--train-steps',type=int,default=1000)
    p.add_argument('--teacher-steps',type=int,default=1000)
    p.add_argument('--finetune-steps',type=int,default=0,help='Optional Meta reward-drift fine-tuning')
    p.add_argument('--sample-steps',type=int,default=8)
    p.add_argument('--posterior-steps',type=int,default=4)
    p.add_argument('--particles',type=int,default=32)
    p.add_argument('--reward-strength',type=float,default=1.)
    p.add_argument('--ssfm-target',choices=['official-code','paper'],default='official-code')
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--samples',type=int,default=128)
    p.add_argument('--width',type=int,default=64)
    p.add_argument('--lr',type=float,default=1e-3)
    p.add_argument('--seed',type=int,default=6270)
    p.add_argument('--threads',type=int,default=1)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--data',help='Whitespace-separated text, one sequence per line; or continuous CSV with two numeric columns')
    p.add_argument('--out',help='Run directory; defaults to lecture_6/outputs/METHOD')
    return p


def restore_model(checkpoint,device):
    method=checkpoint['method'];width=checkpoint['width']
    if method in CATEGORICAL:
        model=SequenceNet(checkpoint['length'],len(checkpoint['vocab']),width)
    elif method=='expanding':
        model=ExpandingNet(checkpoint['length'],len(checkpoint['vocab']),width)
    elif method=='ssfm':
        model=StrongMap(width,checkpoint['sigma'])
    else:
        model=MapNet(checkpoint['dim'],width,3 if method in ['diamond','meta'] else 0)
    model.load_state_dict(checkpoint['model'])
    model=model.to(device).eval()
    return model


def generate(model,state,args):
    """All metrics use fresh, seeded samples; no training examples stand in for output."""
    seed_all(args.seed+100,args.threads)
    method=state['method'];device=args.device
    report={'method':method,'sample_steps':args.sample_steps,'samples':args.samples,
            'sampling_seed':args.seed+100}
    if method in CATEGORICAL:
        ids=sample_categorical(model,args.samples,args.sample_steps,device).cpu()
        texts,metrics=text_metrics(ids,state['vocab'],reference=set(state['reference_text']))
        report.update(metrics)
        return texts,report
    if method=='expanding':
        ids,lengths,extra=sample_expanding(model,args.samples,args.sample_steps,device)
        texts,metrics=text_metrics(ids.cpu(),state['vocab'],lengths.cpu(),set(state['reference_text']))
        report.update(metrics);report.update(extra)
        return texts,report
    if method=='ssfm':
        x,extra=evaluate_stochastic(model,args.samples,args.sample_steps,device)
        report.update(extra)
    elif method in ['diamond','meta']:
        with torch.no_grad():
            outer=torch.randn(args.samples,2,device=device)
            t=torch.zeros(args.samples,1,device=device)
            x=posterior_samples(model,outer,t,1,args.posterior_steps)[:,0]
            guided=guided_samples(model,args.samples,args.sample_steps,args.particles,
                                  args.reward_strength,args.posterior_steps)
            report.update(posterior_diagnostics(model,args.posterior_steps))
            report['unconditional_mean_reward']=float(reward(x,args.reward_strength).mean())
            report['guided_mean_reward']=float(reward(guided,args.reward_strength).mean())
            report['guided_distribution']=mixture_metrics(guided)
            report['posterior_steps']=args.posterior_steps
            report['particles']=args.particles
            observation=torch.tensor([[.5,-.3]],device=device)
            _,estimate,ess=weighted_diamond_samples(observation,.5,4096)
            report['importance_posterior_mean']=estimate.tolist()
            report['importance_effective_sample_size']=ess.tolist()
        # Verify differentiability through the learned posterior's context.
        observation=observation.detach().requires_grad_(True)
        value,_,_=posterior_value(model,observation,.5,args.particles,args.reward_strength,args.posterior_steps)
        gradient=torch.autograd.grad(value.sum(),observation)[0]
        report['posterior_value_context_gradient']=gradient.tolist()
        report['finite_value_gradient']=bool(torch.isfinite(gradient).all())
        report['guided_samples']=guided.detach().cpu().tolist()
        if 'reward_drift' in state:
            drift=MapNet(2,state['width']).to(device)
            drift.load_state_dict(state['reward_drift'])
            # A complete sampler for the fitted drift; boundary-time queries
            # extrapolate beyond the [.05,.85] fine-tuning interval.
            aligned=integrate_velocity(lambda z,a:drift(z,a,a),outer,0.,1.,args.sample_steps)
            report['finetuned_mean_reward']=float(reward(aligned,args.reward_strength).mean())
            report['finetuned_finite_samples']=bool(torch.isfinite(aligned).all())
            report['finetuned_samples']=aligned.detach().cpu().tolist()
            report['finetuned_sampling_note']='Full-interval Heun integration; boundary times extrapolate beyond fine-tuning support.'
    else:
        noise=torch.randn(args.samples,state['dim'],device=device)
        x=sample_continuous(model,method,noise,args.sample_steps)
        if method=='latent':
            ae=Autoencoder(state['width']).to(device)
            ae.load_state_dict(state['autoencoder'])
            with torch.no_grad():
                x=ae.decoder(x*state['latent_std'].to(device)+state['latent_mean'].to(device))
                heldout=state['heldout_data'].to(device)
                reconstruction=ae.decoder(ae.encoder(embed_surface(heldout)))
                report['heldout_reconstruction_mse']=float((reconstruction-embed_surface(heldout)).square().mean())
                report['surface_residual_rmse']=float((x[:,2:]-.3*(x[:,:1].square()-x[:,1:2].square())).square().mean().sqrt())
        if method not in ['consistency','meanflow','flow-matching']:
            from common import finite_map
            with torch.no_grad():
                direct=finite_map(model,noise,0.,1.)
                split=finite_map(model,finite_map(model,noise,0.,.5),.5,1.)
                report['composition_rmse']=float((direct-split).square().mean().sqrt())
    report['finite_samples']=bool(torch.isfinite(x).all())
    if not report['finite_samples']:raise FloatingPointError('Sampling produced nonfinite coordinates.')
    if x.shape[1]>=2 and state['data_kind']=='four-gaussian-mixture':
        report.update(mixture_metrics(x[:,:2]))
    rows=[' '.join(f'{value:.7f}' for value in row) for row in x.detach().cpu().tolist()]
    return rows,report


def main(argv=None):
    p=parser();args=p.parse_args(argv)
    if min(args.train_steps,args.teacher_steps,args.sample_steps,args.posterior_steps,
           args.particles,args.batch_size,args.samples,args.width,args.threads)<1 or args.finetune_steps<0:
        p.error('Step counts, sizes, and width must be positive; finetune steps must be nonnegative.')
    if args.lr<=0:p.error('Learning rate must be positive.')
    if args.device=='cuda' and not torch.cuda.is_available():p.error('CUDA is unavailable; select cpu.')
    if args.mode=='sample' and args.out is None:p.error('Sample mode requires --out for its checkpoint.')
    if args.mode!='sample' and args.method is None:args.method='flow-matching'
    out=Path(args.out) if args.out else ROOT/'outputs'/args.method
    out.mkdir(parents=True,exist_ok=True)
    seed_all(args.seed,args.threads)
    start=time.perf_counter()
    if args.mode=='sample':
        state=torch.load(out/'checkpoint.pt',map_location=args.device,weights_only=True)
        if args.method is not None and args.method!=state['method']:
            p.error('Requested method differs from the saved checkpoint.')
        args.method=state['method']
        if args.method=='ssfm' and args.sample_steps & (args.sample_steps-1):
            p.error('SSFM sample steps must be a power of two.')
        model=restore_model(state,args.device)
        rows,report=generate(model,state,args)
        (out/'resampled.txt').write_text('\n'.join(rows)+'\n')
        write_json(out/'sample_report.json',report)
        print(json.dumps({'method':args.method,'mode':'sample','out':str(out),'seconds':time.perf_counter()-start}))
        return report
    if args.method=='ssfm' and args.sample_steps & (args.sample_steps-1):
        p.error('SSFM sample steps must be a power of two.')
    if args.finetune_steps and args.method!='meta':p.error('--finetune-steps applies to meta.')
    if args.data and args.method in ['diamond','meta','ssfm']:
        p.error('Analytic posterior/OU examples use their specified reference distributions.')
    stage_logs=[]
    if args.method in CATEGORICAL+['expanding']:
        default='variable_text.txt' if args.method=='expanding' else 'phrases.txt'
        data_path=Path(args.data) if args.data else ROOT/'data'/default
        ids,lengths,vocab=load_text(data_path,args.method=='expanding')
        perm=torch.randperm(len(ids));ids,lengths=ids[perm],lengths[perm]
        split=max(1,int(.8*len(ids)))
        training=ids[:split].to(args.device)
        if args.method=='expanding':
            model,state,logs=train_expanding(training,lengths[:split].to(args.device),vocab,args)
        else:
            model,state,logs=train_categorical(args.method,training,vocab,args)
        reference,_=text_metrics(ids[:split],vocab,lengths[:split])
        state.update({'reference_text':reference,'data_kind':'synthetic-text' if args.data is None else 'custom-text'})
        with torch.no_grad():
            if args.method in CATEGORICAL:
                heldout=ids[split:].to(args.device)
                clean=F.one_hot(heldout,len(vocab)).float();t=torch.full((len(clean),1),.5,device=args.device)
                x,_=interpolate(clean,t)
                state['validation_ce']=float(F.cross_entropy(model(x,t,t).flatten(0,1),heldout.flatten()))
    elif args.method=='ssfm':
        model,state,logs=train_stochastic(args,args.device)
        state['data_kind']='ornstein-uhlenbeck'
    else:
        if args.data:
            import numpy as np
            data=torch.tensor(np.loadtxt(args.data,delimiter=','),dtype=torch.float32)
            if data.ndim!=2 or data.shape[1]!=2 or len(data)<8 or not torch.isfinite(data).all():
                p.error('Continuous CSV needs at least eight finite rows and exactly two columns, without a header.')
            data=data[torch.randperm(len(data))];kind='custom-continuous'
        else:
            data=mixture_data(4096);kind='four-gaussian-mixture'
        split=int(.8*len(data));train=data[:split].to(args.device)
        if args.method=='latent':
            model,_,state,logs,stage_logs=train_latent(train,args)
        elif args.method in ['diamond','meta']:
            model,state,logs=train_posterior(args.method,train,args)
            if args.finetune_steps:
                drift,finetune_logs=train_reward_drift(model,train,args)
                state['reward_drift']=drift.state_dict()
                write_json(out/'finetune_losses.json',finetune_logs)
        else:
            model,state,logs,stage_logs=train_continuous(args.method,train,args)
        state.update({'data_kind':kind,'heldout_data':data[split:]})
    state.update({'format_version':1,'method':args.method,'width':args.width,'seed':args.seed})
    torch.save(state,out/'checkpoint.pt')
    write_json(out/'config.json',vars(args))
    write_json(out/'losses.json',logs)
    if stage_logs:write_json(out/'teacher_losses.json',stage_logs)
    report={'method':args.method,'data_kind':state['data_kind'],
            'train_loss_first_20_mean':sum(x['loss'] for x in logs[:20])/len(logs[:20]),
            'train_loss_last_20_mean':sum(x['loss'] for x in logs[-20:])/len(logs[-20:]),
            'training_steps':args.train_steps,'python':platform.python_version(),'torch':str(torch.__version__)}
    if 'validation_ce' in state:report['heldout_diagonal_ce_at_half_time']=state['validation_ce']
    if args.mode=='train-sample':
        model.eval()
        rows,metrics=generate(model,state,args)
        (out/'samples.txt').write_text('\n'.join(rows)+'\n')
        report.update(metrics)
    report['elapsed_seconds']=time.perf_counter()-start
    write_json(out/'report.json',report)
    print(json.dumps({'method':args.method,'out':str(out),'loss':report['train_loss_last_20_mean'],
                      'seconds':report['elapsed_seconds']}),flush=True)
    return report


if __name__=='__main__':main()
