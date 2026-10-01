#!/usr/bin/env python3
"""Run every complete training example; pass --quick for a small CPU check."""
import argparse
from run import main
parser = argparse.ArgumentParser()
parser.add_argument('--quick', action='store_true')
args = parser.parse_args()
methods = ['gat', 'dirichlet', 'fisher', 'gumbel', 'rectified', 'redi', 'mog-dfm', 'areuredi']
for method in methods:
    command = ['--method', method, '--out', 'outputs/' + method]
    if args.quick:
        command += ['--train-steps', '20', '--samples', '4', '--length', '4',
                    '--batch-size', '8', '--sample-steps', '20']
        command += ['--teacher-steps', '20', '--pairs', '32', '--refine-steps', '12']
    main(command)
