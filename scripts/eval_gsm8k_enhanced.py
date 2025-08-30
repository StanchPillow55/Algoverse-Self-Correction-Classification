#!/usr/bin/env python3
import argparse, os, json, glob, csv
from datetime import datetime
from collections import Counter

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True, help='Path to runs directory or JSON file')
    ap.add_argument('--out', default='results/gsm8k_metrics.csv')
    args = ap.parse_args()
    
    os.makedirs('results', exist_ok=True)
    
    # Prepare to write CSV
    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset', 'split', 'model', 'temperature', 'metric_name', 'metric_value', 'timestamp'])
        
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
                
                # Write basic EM score
                writer.writerow([
                    'gsm8k',
                    os.getenv('DATASET_SPLIT', 'unknown'),
                    os.getenv('OPENAI_MODEL', 'gpt-4o-mini'),
                    os.getenv('OPENAI_TEMPERATURE', '0.2'),
                    'EM',
                    f'{mean_acc:.3f}',
                    datetime.utcnow().isoformat() + 'Z'
                ])
                
                # Collect diagnosis information
                traces = data.get('traces', [])
                diagnosis_counts = Counter()
                
                for trace in traces:
                    for turn in trace.get('turns', []):
                        exec_details = turn.get('execution_details', {})
                        diag = exec_details.get('diagnosis', 'unknown')
                        diagnosis_counts[diag] += 1
                
                # Write diagnosis breakdown
                total_attempts = sum(diagnosis_counts.values())
                if total_attempts > 0:
                    for diag, count in diagnosis_counts.most_common():
                        writer.writerow([
                            'gsm8k',
                            os.getenv('DATASET_SPLIT', 'unknown'),
                            os.getenv('OPENAI_MODEL', 'gpt-4o-mini'),
                            os.getenv('OPENAI_TEMPERATURE', '0.2'),
                            f'diagnosis_{diag}',
                            f'{count/total_attempts:.3f}',
                            datetime.utcnow().isoformat() + 'Z'
                        ])
                
            except Exception as e:
                print(f"Error processing {run_file}: {e}")
                continue
    
    print(f"Wrote metrics to {args.out}")

if __name__ == '__main__':
    main()
