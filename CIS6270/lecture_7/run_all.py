#!/usr/bin/env python3
"""Run every example and verify exact seeded checkpoint reloads."""
import argparse
from pathlib import Path
from run import main as run, METHODS, ROOT, write_json

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--quick',action='store_true')
    p.add_argument('--out',default=str(ROOT/'outputs'))
    p.add_argument('--samples',type=int,default=128)
    args=p.parse_args()
    receipts=[]
    for method in METHODS:
        out=Path(args.out)/method
        first=run(['--method',method,'--out',str(out),'--samples',str(args.samples)]+(['--quick'] if args.quick else []))
        second=run(['--mode','sample','--out',str(out),'--samples',str(args.samples)])
        if first != second:
            raise AssertionError('Seeded checkpoint reload differs for '+method)
        receipts.append({'method':method,'checkpoint_reload_equal':True,
                         'finite_values':first['report']['finite_values']})
    write_json(Path(args.out)/'run_all_report.json',{'quick':args.quick,'methods':receipts})
    print(str(len(receipts))+' examples passed checkpoint reload verification.')

if __name__=='__main__':
    main()
