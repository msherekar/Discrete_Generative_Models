#!/usr/bin/env python3
"""Solve or train, save, reload, and sample the Lecture 7 methods."""
import argparse
import inspect
import json
import platform
import time
from pathlib import Path
import torch
import numpy as np
from ot_sbm_examples import (serial, transport_example, finite_bridge_example,
    discrete_imf_example, ctmc_bridge_example, gaussian_example,
    reward_bridge_example, branching_example, entangled_geometry_example)
from learned_bridges import METHODS as LEARNED, save_artifact
from discrete_learning import fit_ddsbm, fit_csbm
from sampling import generate

ROOT = Path(__file__).resolve().parent
EXACT = {'ot': transport_example, 'sinkhorn': transport_example,
         'finite-sb': finite_bridge_example, 'discrete-imf': discrete_imf_example,
         'ctmc-sb': ctmc_bridge_example, 'gaussian-sb': gaussian_example,
         'reward-tilt': reward_bridge_example, 'branch-mass': branching_example,
         'cone-geometry': entangled_geometry_example}
LEARNED = {**LEARNED, 'ddsbm': fit_ddsbm, 'csbm': fit_csbm}
METHODS = list(EXACT) + list(LEARNED)
SEEDS = dict(dsb=7, dsbm=8, sf2m=9, tr2d2=10, branch=11, entangled=12, ddsbm=13, csbm=14)

def write_json(path, value):
    Path(path).write_text(json.dumps(serial(value), indent=2, allow_nan=False)+'\n')

def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method', choices=METHODS)
    p.add_argument('--mode', choices=['train-sample', 'train', 'sample'], default='train-sample')
    p.add_argument('--out', help='Run directory, default lecture_7/outputs/METHOD')
    p.add_argument('--seed', type=int, help='Training seed; sampling uses this seed plus 100')
    p.add_argument('--samples', type=int, default=256)
    p.add_argument('--sample-steps', type=int, help='Override the inference grid when the sampler supports it')
    p.add_argument('--train-steps', type=int, help='Gradient updates for sf2m, branch, ddsbm, or csbm')
    p.add_argument('--rounds', type=int, help='Forward/reverse fitting cycles for dsb or dsbm')
    p.add_argument('--epochs', type=int, help='Replay/CE epochs for tr2d2 or entangled')
    p.add_argument('--batch-size', type=int, help='Training batch size for methods with a batch argument')
    p.add_argument('--searches', type=int, help='MCTS rollouts per TR2-D2 epoch')
    p.add_argument('--quick', action='store_true', help='Short execution check; not a quality experiment')
    return p

def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    for name in ['samples','sample_steps','train_steps','rounds','epochs','batch_size','searches']:
        value = getattr(args, name)
        if value is not None and value < 1:
            p.error(name.replace('_','-')+' must be positive')
    if args.mode == 'sample' and args.out is None:
        p.error('Sample mode requires --out for its checkpoint.')
    if args.mode != 'sample' and args.method is None:
        args.method = 'sinkhorn'
    out = Path(args.out) if args.out else ROOT/'outputs'/args.method
    out.mkdir(parents=True, exist_ok=True)
    began = time.perf_counter()
    if args.mode == 'sample':
        state = torch.load(out/'checkpoint.pt', map_location='cpu', weights_only=True)
        if args.method is not None and args.method != state['method']:
            p.error('The requested method differs from the saved checkpoint.')
        seed = args.seed if args.seed is not None else json.loads((out/'config.json').read_text())['seed']
    else:
        seed = args.seed if args.seed is not None else SEEDS.get(args.method, 6270)
        config = {'method': args.method, 'seed': seed, 'mode': args.mode,
                  'quick': args.quick, 'device': 'cpu', 'python': platform.python_version(),
                  'torch': torch.__version__, 'numpy': np.__version__}
        if args.method in EXACT:
            if any(x is not None for x in [args.train_steps,args.rounds,args.epochs,args.batch_size,args.searches]):
                p.error('This method is a numerical solver; it has no neural training parameters.')
            report = EXACT[args.method]()
            steps = 2 if args.method == 'finite-sb' else 100
            save_artifact(out/'checkpoint.pt', method=args.method, result=serial(report), steps=steps)
            config['training'] = 'Exact or explicitly enumerated numerical calculation'
        else:
            fn = LEARNED[args.method]
            signature = inspect.signature(fn).parameters
            settings = {'seed': seed, 'checkpoint': out/'checkpoint.pt'}
            if args.quick:
                for name, value in dict(updates=20, rounds=2, epochs=1, batch=256, searches=40).items():
                    if name in signature:
                        settings[name] = value
            for arg, param in [('train_steps','updates'),('rounds','rounds'),('epochs','epochs'),('batch_size','batch'),('searches','searches')]:
                value = getattr(args, arg)
                if value is not None:
                    if param not in signature:
                        p.error('--'+arg.replace('_','-')+' does not apply to '+args.method)
                    settings[param] = value
            config['training'] = {k: settings.get(k, v.default) for k,v in signature.items() if k != 'checkpoint'}
            report = fn(**settings)
        write_json(out/'config.json', config)
        write_json(out/'report.json', report)
        write_json(out/'losses.json', report.get('loss_samples', report.get('history', [])))
        state = torch.load(out/'checkpoint.pt', map_location='cpu', weights_only=True)
        if args.mode == 'train':
            print(json.dumps({'method': args.method, 'mode': args.mode, 'out': str(out), 'seconds': time.perf_counter()-began}))
            return report
    output = generate(state, args.samples, seed+100, args.sample_steps)
    stem = 'resampled' if args.mode == 'sample' else 'samples'
    write_json(out/(stem+'.json'), output)
    (out/(stem+'.txt')).write_text('\n'.join(json.dumps(row) for row in output['values'])+'\n')
    write_json(out/('sample_report.json' if args.mode == 'sample' else 'generation_report.json'), output['report'])
    print(json.dumps({'method':state['method'],'mode':args.mode,'out':str(out),'seconds':round(time.perf_counter()-began,3)}),flush=True)
    return output

if __name__ == '__main__':
    main()
