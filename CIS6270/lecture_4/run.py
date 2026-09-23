#!/usr/bin/env python3
"""Train and sample each Lecture 4 method on small, explicit DNA examples."""
import argparse
from pathlib import Path
import torch
from lecture_core import DNA, mdlm_loss, mdlm_sample, udlm_loss, block_loss
from common import (seed_all, load_data, optimize, save_run, ConditionalDNA,
                    decode, metrics)
from diffusion import (uniform_sample, block_sample, conditional_loss,
                       cfg_sample, NoisyClassifier, fit_classifier,
                       classifier_sample, peptune_search)

METHODS = ['mdlm', 'udlm', 'block', 'cfg', 'classifier-free',
           'classifier-gradient', 'classifier-exact', 'peptune']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=METHODS, default='mdlm')
    parser.add_argument('--mode', choices=['train-sample', 'train', 'sample'], default='train-sample')
    parser.add_argument('--train-steps', type=int, default=300)
    parser.add_argument('--sample-steps', type=int, default=40)
    parser.add_argument('--classifier-steps', type=int, default=300)
    parser.add_argument('--search-steps', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--samples', type=int, default=32)
    parser.add_argument('--length', type=int, default=8)
    parser.add_argument('--width', type=int, default=32)
    parser.add_argument('--block-size', type=int, default=2)
    parser.add_argument('--strength', type=float, default=1.5)
    parser.add_argument('--label', type=int, choices=[0, 1], default=1)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--data', help='TSV with sequence and optional binary label columns')
    parser.add_argument('--out', default='outputs/mdlm')
    args = parser.parse_args(argv)
    if min(args.length, args.sample_steps, args.batch_size, args.samples, args.block_size) < 1:
        parser.error('Lengths, step counts, and batch counts must be positive.')
    if args.width % 4 or args.length > 64:
        parser.error('Width must be divisible by four; length must not exceed 64.')
    if args.mode != 'sample' and args.train_steps < 1:
        parser.error('Training requires at least one step.')
    seed_all(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    method = 'cfg' if args.method == 'classifier-free' else args.method
    model = ConditionalDNA(args.width) if method == 'cfg' else DNA(args.width)
    classifier = None
    losses = []
    report = {'method': method, 'data': 'synthetic DNA; not biological validation'}
    if args.mode == 'sample':
        checkpoint = torch.load(out / 'checkpoint.pt', weights_only=True)
        if checkpoint['method'] != method or checkpoint['length'] != args.length:
            raise ValueError('Checkpoint method/length must match command arguments.')
        model.load_state_dict(checkpoint['model'])
        if 'classifier' in checkpoint:
            classifier = NoisyClassifier(args.length)
            classifier.load_state_dict(checkpoint['classifier'])
    else:
        data, labels = load_data(args.data, args.length)
        split = max(1, int(.8 * len(data)))
        train, train_labels = data[:split], labels[:split]
        if method == 'udlm':
            loss_fn = lambda m, x, idx: udlm_loss(m, x)
        elif method == 'block':
            starts = list(range(0, args.length, args.block_size))
            def loss_fn(m, x, idx):
                start = starts[int(torch.randint(len(starts), ()))]
                return len(starts) * block_loss(m, x, start, args.block_size)
        elif method == 'cfg':
            loss_fn = lambda m, x, idx: conditional_loss(m, x, train_labels[idx])
        else:
            loss_fn = lambda m, x, idx: mdlm_loss(m, x)
        losses = optimize(model, loss_fn, train, args.train_steps, args.batch_size)
        checkpoint = {'model': model.state_dict(), 'method': method,
                      'length': args.length, 'width': args.width}
        if method.startswith('classifier-'):
            classifier = NoisyClassifier(args.length)
            classifier_losses = fit_classifier(classifier, train, train_labels,
                                               args.classifier_steps, args.batch_size)
            checkpoint['classifier'] = classifier.state_dict()
            report['classifier_final_loss'] = classifier_losses[-1]
        torch.save(checkpoint, out / 'checkpoint.pt')
        report['train_loss_first_20_mean'] = sum(losses[:20]) / len(losses[:20])
        report['train_loss_last_20_mean'] = sum(losses[-20:]) / len(losses[-20:])
        # An independent noisy validation estimate, not a perplexity claim.
        val = data[split:]
        if len(val):
            with torch.no_grad():
                if method == 'udlm':
                    validation = udlm_loss(model, val)
                elif method == 'cfg':
                    validation = conditional_loss(model, val, labels[split:], drop=0.)
                elif method == 'block':
                    validation = sum(block_loss(model, val, start, args.block_size)
                                     for start in starts)
                else:
                    validation = mdlm_loss(model, val)
            report['validation_loss_one_mc_draw'] = float(validation)
        if args.mode == 'train':
            save_run(out, vars(args), losses, train[:args.samples],
                     {**report, 'sample_file_contains': 'training examples; generation not requested'})
            return report
    model.eval()
    if method == 'udlm':
        samples = uniform_sample(model, args.samples, args.length, args.sample_steps)
        report['endpoint_approximation'] = 't in [0.02, 0.98]; stop at residual noise 0.02, without a posterior interpretation of the UDLM parameter vector'
    elif method == 'block':
        samples = block_sample(model, args.samples, args.length, args.block_size, args.sample_steps)
    elif method == 'cfg':
        samples = cfg_sample(model, args.samples, args.length, args.strength, args.label, args.sample_steps)
    elif method.startswith('classifier-'):
        classifier.eval()
        samples = classifier_sample(model, classifier, args.samples, args.length,
                                    args.strength, args.label, args.sample_steps,
                                    gradient=method == 'classifier-gradient')
        report['guidance'] = 'learned noisy classifier; final residual masks use denoiser closure'
    elif method == 'peptune':
        samples, scores, trace = peptune_search(model, args.length, args.search_steps)
        report['archive_scores'] = scores.tolist()
        report['search_trace'] = trace
        report['scope'] = 'DNA MCTS mechanism; not peptide-model training or paper reproduction'
    else:
        samples = mdlm_sample(model, args.samples, args.length, args.sample_steps)
    return save_run(out, vars(args), losses, samples, report)


if __name__ == '__main__':
    main()
