#!/usr/bin/env python3
"""Run every complete training example; pass --quick for a small CPU check."""
import argparse
from run import main
parser = argparse.ArgumentParser()
parser.add_argument('--quick', action='store_true')
args = parser.parse_args()
methods = ['mdlm', 'udlm', 'block', 'cfg', 'classifier-exact', 'classifier-gradient', 'peptune']
for method in methods:
    command = ['--method', method, '--out', 'outputs/' + method]
    if args.quick:
        command += ['--train-steps', '20', '--samples', '4', '--length', '4',
                    '--batch-size', '8', '--sample-steps', '20']
        command += ['--classifier-steps', '20', '--search-steps', '12']
    main(command)
