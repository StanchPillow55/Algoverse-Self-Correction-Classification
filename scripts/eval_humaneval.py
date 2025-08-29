#!/usr/bin/env python3
import argparse, os, json, glob

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', default='runs/humaneval')
    ap.add_argument('--out', default='results/humaneval_metrics.csv')
    args = ap.parse_args()
    os.makedirs('results', exist_ok=True)
    rows = [['dataset','split','model','temperature','k','metric','value','timestamp']]
    for f in glob.glob(os.path.join(args.runs, '*.json')):
        try:
            d = json.load(open(f))
            val = d.get('summary',{}).get('final_accuracy_mean',0.0)
            rows.append(['humaneval','auto',os.getenv('OPENAI_MODEL','gpt-4o-mini'),os.getenv('OPENAI_TEMPERATURE','0.2'),os.getenv('PASS_K','1'),'pass@k',val,os.getenv('RUN_TS','')])
        except Exception:
            continue
    with open(args.out,'w',encoding='utf-8') as fh:
        for r in rows:
            fh.write(','.join(map(str,r))+'\n')

if __name__ == '__main__':
    main()

