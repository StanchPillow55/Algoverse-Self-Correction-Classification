#!/usr/bin/env python3
import argparse, os, json
from src.metrics.accuracy import gsm8k_em
from src.utils.dataset_loader import read_csv_flexible
from src.agents.learner import LearnerBot

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=os.getenv('OPENAI_MODEL','gpt-4o-mini'))
    ap.add_argument('--temperature', type=float, default=0.2)
    ap.add_argument('--dataset', default='data/math_sample_20.csv')
    ap.add_argument('--out', default='runs/gsm8k/run.json')
    ap.add_argument('--provider', default=os.getenv('PROVIDER','openai'))
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.environ['OPENAI_TEMPERATURE'] = str(args.temperature)
    bot = LearnerBot(provider=args.provider, model=args.model)

    df = read_csv_flexible(args.dataset)
    items = df.to_dict(orient='records')
    traces = []
    correct = 0
    for i, row in enumerate(items):
        q = row.get('question') or row.get('prompt') or ''
        ref = row.get('ground_truth') or row.get('answer') or ''
        prompt = (
            "You are a meticulous math solver. Think privately. Provide only the final numeric answer.\n\n"
            "Solve the problem. Think silently and provide only the final numeric answer with no units.\n\nQuestion:\n" + q + "\n\nOutput format: a single line containing only the final number."
        )
        a, _ = bot.answer(prompt, [], template=None)
        acc = gsm8k_em(a, ref)
        correct += acc
        traces.append({'qid': row.get('id', f'q{i+1}'), 'question': q, 'reference': ref, 'answer': a, 'accuracy': acc})

    out = {'summary': {'items': len(traces), 'final_accuracy_mean': (correct/max(1,len(traces)))} , 'traces': traces}
    with open(args.out,'w',encoding='utf-8') as f:
        json.dump(out, f, indent=2)

if __name__ == '__main__':
    main()

