#!/usr/bin/env python3
"""
Re-evaluate GSM8K runs with the fixed evaluator.
"""

import json
import pandas as pd
import glob
from pathlib import Path
from src.metrics.accuracy import gsm8k_em, gsm8k_extract_gold_answer, normalize_numeric_string

def reevaluate_gsm8k_run(json_file_path: str) -> dict:
    """Re-evaluate a single GSM8K run with the fixed evaluator."""
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    traces = data.get('traces', [])
    if not traces:
        return {'accuracy': 0.0, 'correct': 0, 'total': 0, 'file': json_file_path}
    
    correct = 0
    total = len(traces)
    
    # Re-evaluate each trace
    for trace in traces:
        reference = trace.get('reference', '')
        turns = trace.get('turns', [])
        
        if not turns:
            continue
            
        # Get the final answer from the last turn
        final_turn = turns[-1]
        answer = str(final_turn.get('answer', ''))
        
        # Compute EM with fixed evaluator
        em_result = gsm8k_em(answer, reference)
        correct += em_result
        
        # Update the trace with corrected accuracy
        trace['final_accuracy'] = em_result
        for turn in turns:
            turn['accuracy'] = em_result  # Update all turns for consistency
    
    # Update summary
    final_accuracy = correct / total if total > 0 else 0.0
    data['summary']['final_accuracy_mean'] = final_accuracy
    
    # Save updated data back to file
    with open(json_file_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    return {
        'accuracy': final_accuracy,
        'correct': correct,
        'total': total,
        'file': json_file_path
    }

def main():
    """Re-evaluate all GSM8K runs."""
    base_dir = Path("/Users/bradleyharaguchi/Algoverse-Self-Correction-Classification")
    runs_dir = base_dir / "runs"
    
    # Find all GSM8K JSON files
    gsm8k_files = []
    gsm8k_files.extend(runs_dir.glob("full/gsm8k*.json"))
    gsm8k_files.extend(runs_dir.glob("experiments/*/gsm8k*.json"))
    
    # Filter out test files
    gsm8k_files = [f for f in gsm8k_files if '20.json' not in str(f) and 'traces.json' not in str(f)]
    
    print(f"Found {len(gsm8k_files)} GSM8K run files to re-evaluate:")
    
    results = []
    for json_file in gsm8k_files:
        print(f"  Re-evaluating {json_file}")
        result = reevaluate_gsm8k_run(str(json_file))
        results.append(result)
        print(f"    {result['correct']}/{result['total']} correct ({result['accuracy']:.3f} accuracy)")
    
    print(f"\nRe-evaluation complete!")
    print(f"Results summary:")
    for result in results:
        file_name = Path(result['file']).name
        print(f"  {file_name}: {result['accuracy']:.3f} ({result['correct']}/{result['total']})")
    
    # Overall statistics
    total_correct = sum(r['correct'] for r in results)
    total_problems = sum(r['total'] for r in results)
    overall_accuracy = total_correct / total_problems if total_problems > 0 else 0.0
    
    print(f"\nOverall: {total_correct}/{total_problems} = {overall_accuracy:.3f} accuracy")

if __name__ == "__main__":
    main()
