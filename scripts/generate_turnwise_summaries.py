#!/usr/bin/env python3
"""
Generate per-turn summaries from trace data for HumanEval and GSM8K datasets.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob
from typing import Dict, List, Tuple

def parse_humaneval_traces(base_dir: Path) -> pd.DataFrame:
    """Parse HumanEval traces from JSON files or JSONL files."""
    trace_data = []
    
    # Look for full run JSON files first
    json_files = list(base_dir.glob("full/heval_full*.json")) + \
                 list(base_dir.glob("full/humaneval*.json"))
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                if 'traces' in data:
                    traces = data['traces']
                    for trace in traces:
                        problem_id = trace.get('problem_id', 'unknown')
                        for turn in trace.get('turns', []):
                            trace_data.append({
                                'dataset': 'humaneval',
                                'problem_id': problem_id,
                                'turn': turn.get('turn_index', 0) + 1,  # 1-indexed
                                'is_correct': turn.get('is_correct', False),
                                'final_correct': trace.get('final_correct', False)
                            })
        except Exception as e:
            print(f"Error parsing {json_file}: {e}")
    
    # Also look for JSONL traces
    jsonl_files = list(base_dir.glob("**/humaneval*/traces.jsonl")) + \
                  list(base_dir.glob("**/heval*/traces.jsonl"))
    
    for jsonl_file in jsonl_files:
        try:
            with open(jsonl_file, 'r') as f:
                for line in f:
                    trace = json.loads(line.strip())
                    problem_id = trace.get('problem_id', 'unknown')
                    for turn in trace.get('turns', []):
                        trace_data.append({
                            'dataset': 'humaneval',
                            'problem_id': problem_id,
                            'turn': turn.get('turn_index', 0) + 1,
                            'is_correct': turn.get('is_correct', False),
                            'final_correct': trace.get('final_correct', False)
                        })
        except Exception as e:
            print(f"Error parsing {jsonl_file}: {e}")
    
    return pd.DataFrame(trace_data)

def parse_gsm8k_traces(base_dir: Path) -> pd.DataFrame:
    """Parse GSM8K traces from JSON files."""
    trace_data = []
    
    # Look for full run JSON files
    json_files = list(base_dir.glob("full/gsm8k*.json")) + \
                 list(base_dir.glob("experiments/*/gsm8k*.json"))
    
    for json_file in json_files:
        # Skip small test files
        if '20.json' in str(json_file) or '100.json' in str(json_file):
            continue
            
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                if 'traces' in data:
                    traces = data['traces']
                    for trace in traces:
                        problem_id = trace.get('qid', trace.get('problem_id', 'unknown'))
                        turns = trace.get('turns', [])
                        final_correct = trace.get('final_accuracy', False)
                        
                        # GSM8K doesn't have turn_index, so enumerate turns
                        for turn_idx, turn in enumerate(turns):
                            trace_data.append({
                                'dataset': 'gsm8k',
                                'problem_id': problem_id,
                                'turn': turn_idx + 1,  # 1-indexed
                                'is_correct': bool(turn.get('accuracy', False)),
                                'final_correct': bool(final_correct)
                            })
        except Exception as e:
            print(f"Error parsing {json_file}: {e}")
    
    return pd.DataFrame(trace_data)

def compute_turnwise_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    """Compute accuracy per turn."""
    if df.empty:
        return pd.DataFrame()
    
    # Group by turn and compute mean accuracy
    turnwise = df.groupby('turn').agg({
        'is_correct': 'mean',
        'problem_id': 'count'
    }).reset_index()
    
    turnwise.columns = ['turn', 'accuracy_mean', 'n_items']
    return turnwise

def save_turnwise_summaries(base_dir: Path):
    """Generate and save turnwise summaries for both datasets."""
    results_dir = base_dir / "results"
    results_dir.mkdir(exist_ok=True)
    
    # Process HumanEval
    print("Processing HumanEval traces...")
    heval_df = parse_humaneval_traces(base_dir / "runs")
    if not heval_df.empty:
        heval_turnwise = compute_turnwise_accuracy(heval_df)
        heval_turnwise['dataset'] = 'humaneval'
        heval_turnwise[['dataset', 'turn', 'accuracy_mean', 'n_items']].to_csv(
            results_dir / "turnwise_humaneval.csv", index=False
        )
        print(f"  Found {len(heval_df)} turn records across {heval_df['problem_id'].nunique()} problems")
        print(f"  Saved turnwise summary to results/turnwise_humaneval.csv")
    else:
        print("  No HumanEval traces found")
    
    # Process GSM8K
    print("\nProcessing GSM8K traces...")
    gsm8k_df = parse_gsm8k_traces(base_dir / "runs")
    if not gsm8k_df.empty:
        gsm8k_turnwise = compute_turnwise_accuracy(gsm8k_df)
        gsm8k_turnwise['dataset'] = 'gsm8k'
        gsm8k_turnwise[['dataset', 'turn', 'accuracy_mean', 'n_items']].to_csv(
            results_dir / "turnwise_gsm8k.csv", index=False
        )
        print(f"  Found {len(gsm8k_df)} turn records across {gsm8k_df['problem_id'].nunique()} problems")
        print(f"  Saved turnwise summary to results/turnwise_gsm8k.csv")
    else:
        print("  No GSM8K traces found")
    
    return heval_turnwise if not heval_df.empty else pd.DataFrame(), \
           gsm8k_turnwise if not gsm8k_df.empty else pd.DataFrame()

def create_turnwise_plot(base_dir: Path):
    """Create turnwise accuracy plot."""
    results_dir = base_dir / "results"
    reports_dir = base_dir / "reports" / "figures"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Load turnwise data
    heval_file = results_dir / "turnwise_humaneval.csv"
    gsm8k_file = results_dir / "turnwise_gsm8k.csv"
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if heval_file.exists():
        heval_df = pd.read_csv(heval_file)
        if not heval_df.empty:
            ax.plot(heval_df['turn'], heval_df['accuracy_mean'], 
                   marker='o', label='HumanEval', linewidth=2)
    
    if gsm8k_file.exists():
        gsm8k_df = pd.read_csv(gsm8k_file)
        if not gsm8k_df.empty:
            ax.plot(gsm8k_df['turn'], gsm8k_df['accuracy_mean'], 
                   marker='s', label='GSM8K', linewidth=2)
    
    ax.set_xlabel('Turn Number', fontsize=12)
    ax.set_ylabel('Average Accuracy', fontsize=12)
    ax.set_title('Accuracy by Turn Number', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, 4))  # Assuming max 3 turns
    
    plt.tight_layout()
    plt.savefig(reports_dir / "turnwise_accuracy.png", dpi=150, bbox_inches='tight')
    print(f"\nSaved turnwise accuracy plot to reports/figures/turnwise_accuracy.png")
    plt.close()

def main():
    """Main execution."""
    base_dir = Path("/Users/bradleyharaguchi/Algoverse-Self-Correction-Classification")
    
    # Generate turnwise summaries
    heval_turnwise, gsm8k_turnwise = save_turnwise_summaries(base_dir)
    
    # Create visualization
    create_turnwise_plot(base_dir)
    
    # Print summary
    if not heval_turnwise.empty:
        print("\nHumanEval Turn-wise Accuracy:")
        print(heval_turnwise.to_string(index=False))
    
    if not gsm8k_turnwise.empty:
        print("\nGSM8K Turn-wise Accuracy:")
        print(gsm8k_turnwise.to_string(index=False))

if __name__ == "__main__":
    main()
