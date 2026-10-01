#!/usr/bin/env python3
"""Train and generate DNA with classic DFM before the simplex extensions."""
import argparse
from pathlib import Path
import torch
from lecture_core import (DNA, gat_loss, gat_sample, dirichlet_loss,
                          fisher_loss, gumbel_loss)
from common import seed_all, load_data, optimize, save_run
from flows import (simplex_sample, rectified_training_loss, make_teacher_pairs,
                   mog_sample, refine_areuredi)

METHODS = ['gat', 'dirichlet', 'fisher', 'gumbel', 'rectified',
           'redi', 'mog-dfm', 'areuredi']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=METHODS, default='gat')
    parser.add_argument('--mode', choices=['train-sample', 'train', 'sample'], default='train-sample')
    parser.add_argument('--train-steps', type=int, default=300)
    parser.add_argument('--teacher-steps', type=int, default=300)
    parser.add_argument('--sample-steps', type=int, default=100)
    parser.add_argument('--refine-steps', type=int, default=100)
    parser.add_argument('--pairs', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--samples', type=int, default=32)
    parser.add_argument('--length', type=int, default=8)
    parser.add_argument('--width', type=int, default=32)
    parser.add_argument('--guidance', type=float, default=0., help='Toy GC objective for simplex samplers')
    parser.add_argument('--strength', type=float, default=1., help='MOG rate multiplier beta')
    parser.add_argument('--preference', type=float, nargs=2, default=[.7, .3])
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--data', help='TSV with DNA sequence column')
    parser.add_argument('--out', default='outputs/gat')
    args = parser.parse_args(argv)
    if min(args.length, args.sample_steps, args.samples, args.batch_size, args.pairs) < 1:
        parser.error('Lengths and sample/batch counts must be positive.')
    if args.width % 4 or args.length > 64:
        parser.error('Width must be divisible by four; length must be at most 64.')
    if min(args.preference) <= 0 or args.strength <= 0:
        parser.error('Preference weights and MOG strength must be positive.')
    if args.mode != 'sample' and min(args.train_steps, args.teacher_steps) < 1:
        parser.error('Training step counts must be positive.')
    preference = [p / sum(args.preference) for p in args.preference]
    seed_all(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    data, labels = load_data(args.data, args.length)
    split = max(1, int(.8 * len(data)))
    train = data[:split]
    model = DNA(args.width)
    losses = []
    report = {'method': args.method, 'data': 'synthetic DNA; not biological validation'}
    loss_fns = {'gat': gat_loss, 'dirichlet': dirichlet_loss,
                'fisher': fisher_loss, 'gumbel': gumbel_loss,
                'rectified': rectified_training_loss, 'mog-dfm': gat_loss}
    if args.mode == 'sample':
        checkpoint = torch.load(out / 'checkpoint.pt', weights_only=True)
        if checkpoint['method'] != args.method or checkpoint['length'] != args.length:
            raise ValueError('Checkpoint method/length must match command arguments.')
        model.load_state_dict(checkpoint['model'])
        if 'reference_data' in checkpoint:
            train = checkpoint['reference_data']
    else:
        if args.method in ['redi', 'areuredi']:
            teacher = DNA(args.width)
            teacher_losses = optimize(teacher, lambda m,x,i: gat_loss(m,x),
                                      train, args.teacher_steps, args.batch_size)
            source, target = make_teacher_pairs(teacher, args.pairs, args.length,
                                                min(args.sample_steps, 100))
            torch.save({'source': source, 'target': target}, out / 'teacher_pairs.pt')
            losses = optimize(model, lambda m,x,i: gat_loss(m,x,source[i]),
                              target, args.train_steps, args.batch_size)
            report['teacher_loss_last_20_mean'] = sum(teacher_losses[-20:]) / len(teacher_losses[-20:])
            report['paired_examples'] = args.pairs
            report['redi_scope'] = 'one teacher-recoupling round; no guaranteed TC reduction'
        else:
            loss_fn = loss_fns[args.method]
            losses = optimize(model, lambda m,x,i: loss_fn(m,x), train,
                              args.train_steps, args.batch_size)
        checkpoint = {'model': model.state_dict(), 'method': args.method,
                      'length': args.length, 'width': args.width}
        if args.method == 'areuredi':
            checkpoint['reference_data'] = train
        torch.save(checkpoint, out / 'checkpoint.pt')
        report['train_loss_first_20_mean'] = sum(losses[:20]) / len(losses[:20])
        report['train_loss_last_20_mean'] = sum(losses[-20:]) / len(losses[-20:])
        if args.method in loss_fns and len(data[split:]):
            with torch.no_grad():
                report['validation_loss_one_mc_draw'] = float(loss_fns[args.method](model, data[split:]))
        if args.mode == 'train':
            return save_run(out, vars(args), losses, train[:args.samples],
                            {**report, 'sample_file_contains': 'training examples; generation not requested'})
    model.eval()
    if args.method in ['dirichlet', 'fisher', 'gumbel', 'rectified']:
        samples, extra = simplex_sample(model, args.method, args.samples,
                                       args.length, args.sample_steps, guidance=args.guidance)
        report.update(extra)
        if args.method == 'gumbel':
            report['gumbel_scope'] = ('beta=1, tau_max=4, decay=4; finite noisy endpoint '
                                     'and approximate data-independent base; no exact data endpoint claim')
    elif args.method == 'mog-dfm':
        samples, extra = mog_sample(model, args.samples, args.length,
                                    args.sample_steps, preference, args.strength)
        report.update(extra)
    else:
        samples = gat_sample(model, args.samples, args.length, args.sample_steps)
        if args.method == 'areuredi':
            samples, extra = refine_areuredi(samples, train, args.refine_steps,
                                            preference, seed=args.seed)
            report.update(extra)
    return save_run(out, vars(args), losses, samples, report)


if __name__ == '__main__':
    main()
