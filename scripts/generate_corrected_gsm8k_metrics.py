#!/usr/bin/env python3
"""
Generate corrected GSM8K metrics CSV from re-evaluated runs.
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime

def extract_ablation_arm(file_path: str) -> str:
    """Extract ablation arm from file path."""
    path_str = str(file_path).lower()
    
    if 'baseline' in path_str:
        return 'baseline'
    elif 'confidence_only' in path_str:
        return 'confidence_only'
    elif 'error_awareness_only' in path_str:
        return 'error_awareness_only'
    elif 'multiturn_only' in path_str:
        return 'multiturn_only'
    elif 'full_system' in path_str:
        return 'full_system'
    elif 'full' in path_str:
        return 'baseline'
    else:
        return 'unknown'

def main():
    """Generate corrected metrics CSV files."""
    base_dir = Path("/Users/bradleyharaguchi/Algoverse-Self-Correction-Classification")
    runs_dir = base_dir / "runs"
    results_dir = base_dir / "results"
    
    # Find all GSM8K JSON files
    gsm8k_files = []
    gsm8k_files.extend(runs_dir.glob("full/gsm8k*.json"))
    gsm8k_files.extend(runs_dir.glob("experiments/*/gsm8k*.json"))
    
    # Filter out test files
    gsm8k_files = [f for f in gsm8k_files if '20.json' not in str(f) and 'traces.json' not in str(f)]
    
    print(f"Processing {len(gsm8k_files)} GSM8K files...")
    
    # Collect metrics
    metrics_data = []
    
    for json_file in gsm8k_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            summary = data.get('summary', {})
            traces = data.get('traces', [])
            accuracy = summary.get('final_accuracy_mean', 0.0)
            num_problems = len(traces)
            ablation_arm = extract_ablation_arm(str(json_file))
            
            is_full_run = num_problems >= 900
            
            metrics_data.append({
                'dataset': 'gsm8k',
                'split': 'auto',
                'model': 'gpt-4o',
                'temperature': '0.2',
                'k': '',
                'metric': 'accuracy',
                'value': accuracy,
                'timestamp': datetime.now().strftime('%Y%m%dT%H%M%S'),
                'num_problems': num_problems,
                'ablation_arm': ablation_arm,
                'run_id': json_file.stem,
                'is_full_run': is_full_run
            })
            
            print(f"  {json_file.name}: {ablation_arm} - {accuracy:.3f} accuracy ({num_problems} problems)")
            
        except Exception as e:
            print(f"  Error processing {json_file}: {e}")
    
    # Convert to DataFrame
    df = pd.DataFrame(metrics_data)
    
    # Save full metrics
    df.to_csv(results_dir / 'gsm8k_metrics_corrected.csv', index=False)
    print(f"\nSaved all metrics to results/gsm8k_metrics_corrected.csv")
    
    # Save full runs only
    full_runs = df[df['is_full_run']]
    full_runs.to_csv(results_dir / 'gsm8k_full_metrics_corrected.csv', index=False)
    print(f"Saved full run metrics to results/gsm8k_full_metrics_corrected.csv")
    
    # Print summary
    print(f"\nSUMMARY:")
    print(f"  Total runs: {len(df)}")
    print(f"  Full runs: {len(full_runs)}")
    if len(full_runs) > 0:
        print(f"  Average accuracy (full runs): {full_runs['value'].mean():.3f}")
    
    print(f"\nFull runs by ablation arm:")
    for arm in sorted(full_runs['ablation_arm'].unique()):
        arm_data = full_runs[full_runs['ablation_arm'] == arm]
        if len(arm_data) > 0:
            print(f"  {arm}: {arm_data['value'].mean():.3f} avg ({len(arm_data)} runs)")

if __name__ == "__main__":
    main()
