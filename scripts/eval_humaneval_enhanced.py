#!/usr/bin/env python3
import argparse, os, json, glob, csv
from datetime import datetime
from src.evaluation.humaneval_evaluator import HumanEvalEvaluator

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True, help='Path to runs directory or JSON file')
    ap.add_argument('--k', nargs='+', type=int, default=[1, 5], help='k values for pass@k')
    ap.add_argument('--out', default='results/heval_metrics.csv')
    args = ap.parse_args()
    
    os.makedirs('results', exist_ok=True)
    
    # Prepare to write CSV
    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset', 'split', 'model', 'temperature', 'k', 'metric_name', 'metric_value', 'timestamp'])
        
        # Find all run files
        if os.path.isfile(args.runs):
            run_files = [args.runs]
        else:
            run_files = glob.glob(os.path.join(args.runs, '*.json'))
        
        for run_file in run_files:
            try:
                with open(run_file, 'r') as rf:
                    data = json.load(rf)
                
                # Extract basic info
                summary = data.get('summary', {})
                mean_acc = summary.get('final_accuracy_mean', 0.0)
                
                # Write basic accuracy
                writer.writerow([
                    'humaneval',
                    os.getenv('DATASET_SPLIT', 'unknown'),
                    os.getenv('OPENAI_MODEL', 'gpt-4o-mini'),
                    os.getenv('OPENAI_TEMPERATURE', '0.2'),
                    '1',
                    'accuracy',
                    f'{mean_acc:.3f}',
                    datetime.utcnow().isoformat() + 'Z'
                ])
                
                # If we have traces, compute pass@k
                traces = data.get('traces', [])
                if traces:
                    # Collect pass/fail results
                    pass_results = []
                    for trace in traces:
                        final_acc = trace.get('final_accuracy', 0)
                        pass_results.append(bool(final_acc))
                    
                    # Compute pass@k for different k values
                    for k in args.k:
                        if k <= len(pass_results):
                            # Simple pass@k: at least one success in first k attempts
                            pass_at_k = float(any(pass_results[:k]))
                            writer.writerow([
                                'humaneval',
                                os.getenv('DATASET_SPLIT', 'unknown'),
                                os.getenv('OPENAI_MODEL', 'gpt-4o-mini'),
                                os.getenv('OPENAI_TEMPERATURE', '0.2'),
                                str(k),
                                f'pass@{k}',
                                f'{pass_at_k:.3f}',
                                datetime.utcnow().isoformat() + 'Z'
                            ])
                
            except Exception as e:
                print(f"Error processing {run_file}: {e}")
                continue
    
    print(f"Wrote metrics to {args.out}")

if __name__ == '__main__':
    main()
