#!/usr/bin/env python3
"""Run every method, then verify generation from each saved checkpoint."""
import argparse
from pathlib import Path

from run import METHODS, ROOT, main


def run_all():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick',action='store_true',help='20-step execution check; not a quality benchmark')
    parser.add_argument('--out',default=str(ROOT/'outputs'))
    parser.add_argument('--train-steps',type=int,default=1000)
    args=parser.parse_args()
    for method in METHODS:
        out=Path(args.out)/method
        command=['--method',method,'--out',str(out)]
        if args.quick:
            command+=['--train-steps','20','--teacher-steps','20','--batch-size','8',
                      '--samples','8','--width','32','--sample-steps','4','--posterior-steps','2','--particles','4']
        else:command+=['--train-steps',str(args.train_steps),'--teacher-steps',str(args.train_steps)]
        main(command)
        sample_args=['--mode','sample','--out',str(out)]
        if args.quick:
            sample_args+=['--samples','8','--sample-steps','4','--posterior-steps','2','--particles','4']
        main(sample_args)
        if (out/'samples.txt').read_bytes()!=(out/'resampled.txt').read_bytes():
            raise AssertionError(f'{method}: seeded generation changed after checkpoint reload')
    print(f'All {len(METHODS)} methods trained, sampled, and reproduced samples after checkpoint reload.')


if __name__=='__main__':run_all()
