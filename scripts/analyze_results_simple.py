#!/usr/bin/env python3
"""
Simple Result Analysis

Analyzes results from scaling experiments without complex imports.
"""

import json
import argparse
from pathlib import Path
import pandas as pd

def analyze_results(results_dir, pattern="*.json"):
    """Analyze results from a directory."""
    results_dir = Path(results_dir)
    
    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        return None
    
    # Find result files
    result_files = list(results_dir.glob(pattern))
    
    if not result_files:
        print(f"❌ No result files found in {results_dir}")
        return None
    
    print(f"📊 Analyzing {len(result_files)} result files")
    
    # Load and analyze results
    results = []
    
    for file_path in result_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Extract basic metrics
            traces = data.get('traces', [])
            if not traces:
                continue
            
            # Calculate accuracy metrics
            initial_correct = sum(1 for t in traces if t.get('turns', [{}])[0].get('accuracy', 0) == 1)
            final_correct = sum(1 for t in traces if t.get('final_accuracy', 0) == 1)
            
            initial_accuracy = initial_correct / len(traces)
            final_accuracy = final_correct / len(traces)
            improvement = final_accuracy - initial_accuracy
            
            # Extract model info from filename
            filename = file_path.stem
            parts = filename.split('_')
            model_name = parts[0] if parts else "unknown"
            
            results.append({
                'file': str(file_path),
                'model': model_name,
                'samples': len(traces),
                'initial_accuracy': initial_accuracy,
                'final_accuracy': final_accuracy,
                'improvement': improvement
            })
            
        except Exception as e:
            print(f"⚠️  Failed to load {file_path}: {e}")
            continue
    
    if not results:
        print("❌ No valid results found")
        return None
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Calculate summary statistics
    summary = {
        'total_experiments': len(results),
        'total_samples': df['samples'].sum(),
        'avg_improvement': df['improvement'].mean(),
        'max_improvement': df['improvement'].max(),
        'min_improvement': df['improvement'].min(),
        'improvement_std': df['improvement'].std()
    }
    
    # Group by model
    by_model = df.groupby('model').agg({
        'samples': 'sum',
        'improvement': ['mean', 'std', 'count']
    }).round(3)
    
    return {
        'summary': summary,
        'by_model': by_model,
        'raw_data': results
    }

def print_summary(analysis):
    """Print analysis summary."""
    if not analysis:
        return
    
    print("\n📊 Scaling Study Analysis Summary")
    print("=" * 50)
    
    summary = analysis['summary']
    print(f"Total experiments: {summary['total_experiments']}")
    print(f"Total samples: {summary['total_samples']}")
    print(f"Average improvement: {summary['avg_improvement']:.3f}")
    print(f"Max improvement: {summary['max_improvement']:.3f}")
    print(f"Min improvement: {summary['min_improvement']:.3f}")
    print(f"Improvement std: {summary['improvement_std']:.3f}")
    
    print("\nBy Model:")
    by_model = analysis['by_model']
    for model in by_model.index:
        mean_imp = by_model.loc[model, ('improvement', 'mean')]
        std_imp = by_model.loc[model, ('improvement', 'std')]
        count = by_model.loc[model, ('improvement', 'count')]
        samples = by_model.loc[model, ('samples', 'sum')]
        
        print(f"  {model:15} | {count:2} exps | {samples:4} samples | {mean_imp:.3f} ± {std_imp:.3f}")

def main():
    parser = argparse.ArgumentParser(description="Analyze scaling study results")
    parser.add_argument("--results-dir", default="outputs/scaling_experiments",
                       help="Directory containing experiment results")
    parser.add_argument("--pattern", default="*.json",
                       help="Pattern to match result files")
    
    args = parser.parse_args()
    
    print("🔍 Analyzing Scaling Study Results")
    print("=" * 40)
    
    # Analyze results
    analysis = analyze_results(args.results_dir, args.pattern)
    
    if analysis:
        print_summary(analysis)
        
        # Save analysis
        output_file = Path(args.results_dir) / "simple_analysis.json"
        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        print(f"\n✅ Analysis saved to: {output_file}")
    else:
        print("❌ Analysis failed")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
