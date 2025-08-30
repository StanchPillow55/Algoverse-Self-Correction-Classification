#!/usr/bin/env python3
import argparse, os, json, glob
import pandas as pd
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='input_dir', default='runs/experiments')
    ap.add_argument('--out', default='reports/ablation_summary.md')
    ap.add_argument('--csv', default='reports/ablation_summary.csv')
    args = ap.parse_args()
    
    os.makedirs('reports', exist_ok=True)
    
    results = []
    
    # Iterate through experiment arms
    arms = ['baseline', 'confidence_only', 'error_awareness_only', 'multiturn_only', 'full_system']
    
    for arm in arms:
        arm_dir = Path(args.input_dir) / arm
        if not arm_dir.exists():
            continue
            
        # Find all result files in this arm
        for json_file in arm_dir.glob('*.json'):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                summary = data.get('summary', {})
                traces = data.get('traces', [])
                
                # Determine dataset from filename
                dataset = 'humaneval' if 'heval' in json_file.name else 'gsm8k'
                
                # Calculate per-turn accuracy
                turn_accuracies = {}
                for trace in traces:
                    for i, turn in enumerate(trace.get('turns', [])):
                        if i not in turn_accuracies:
                            turn_accuracies[i] = []
                        turn_accuracies[i].append(turn.get('accuracy', 0))
                
                # Average per turn
                per_turn_avg = {f'turn_{k}': sum(v)/len(v) if v else 0 for k, v in turn_accuracies.items()}
                
                results.append({
                    'arm': arm,
                    'dataset': dataset,
                    'samples': summary.get('items', 0),
                    'final_accuracy': summary.get('final_accuracy_mean', 0.0),
                    **per_turn_avg
                })
                
            except Exception as e:
                print(f"Error processing {json_file}: {e}")
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save CSV
    df.to_csv(args.csv, index=False)
    
    # Create markdown report
    with open(args.out, 'w') as f:
        f.write("# Ablation Study Results\n\n")
        
        # Summary table by dataset
        for dataset in df['dataset'].unique():
            f.write(f"## {dataset.upper()} Results\n\n")
            
            dataset_df = df[df['dataset'] == dataset]
            
            # Create pivot table
            pivot = dataset_df.pivot_table(
                index='arm',
                values=['final_accuracy'] + [c for c in dataset_df.columns if c.startswith('turn_')],
                aggfunc='mean'
            )
            
            f.write(pivot.to_markdown() + "\n\n")
            
            # Calculate deltas from baseline
            if 'baseline' in dataset_df['arm'].values:
                baseline_acc = dataset_df[dataset_df['arm'] == 'baseline']['final_accuracy'].iloc[0]
                f.write("### Delta from Baseline\n\n")
                f.write("| Arm | Final Accuracy | Delta |\n")
                f.write("|-----|---------------|-------|\n")
                
                for _, row in dataset_df.iterrows():
                    delta = row['final_accuracy'] - baseline_acc
                    sign = '+' if delta >= 0 else ''
                    f.write(f"| {row['arm']} | {row['final_accuracy']:.3f} | {sign}{delta:.3f} |\n")
                
                f.write("\n")
        
        # Overall summary
        f.write("## Overall Summary\n\n")
        f.write("- **Baseline Performance**: Foundation for comparison\n")
        f.write("- **Confidence Only**: Impact of confidence scoring\n")
        f.write("- **Error Awareness Only**: Impact of error detection\n")
        f.write("- **Multiturn Only**: Impact of iterative refinement\n")
        f.write("- **Full System**: Combined effect of all components\n\n")
        
        # Key findings
        f.write("## Key Findings\n\n")
        
        if len(df) > 0:
            best_arm = df.groupby('arm')['final_accuracy'].mean().idxmax()
            best_score = df.groupby('arm')['final_accuracy'].mean().max()
            f.write(f"- Best performing arm: **{best_arm}** with average accuracy {best_score:.3f}\n")
            
            # Check if multiturn helps
            if 'baseline' in df['arm'].values and 'multiturn_only' in df['arm'].values:
                baseline_avg = df[df['arm'] == 'baseline']['final_accuracy'].mean()
                multiturn_avg = df[df['arm'] == 'multiturn_only']['final_accuracy'].mean()
                improvement = multiturn_avg - baseline_avg
                f.write(f"- Multiturn improvement over baseline: {improvement:+.3f}\n")
    
    print(f"Wrote ablation report to {args.out}")
    print(f"Wrote CSV to {args.csv}")

if __name__ == '__main__':
    main()
