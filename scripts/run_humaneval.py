#!/usr/bin/env python3
import argparse, os
from src.loop.runner import run_dataset

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=os.getenv('OPENAI_MODEL','gpt-4o-mini'))
    ap.add_argument('--k', nargs='+', type=int, default=[1])
    ap.add_argument('--temperature', type=float, default=0.2)
    ap.add_argument('--subset', default='subset_20')
    ap.add_argument('--out', default='runs/humaneval/run.json')
    ap.add_argument('--provider', default=os.getenv('PROVIDER','openai'))
    args = ap.parse_args()
    os.environ['OPENAI_TEMPERATURE'] = str(args.temperature)
    for kk in args.k:
        os.environ['PASS_K'] = str(kk)
        run_dataset('humaneval', traces_out=args.out, max_turns=1, provider=args.provider, k=kk, subset=args.subset)

if __name__ == '__main__':
    main()

