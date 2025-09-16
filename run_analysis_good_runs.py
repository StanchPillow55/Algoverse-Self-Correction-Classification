#!/usr/bin/env python3
"""
Wrapper script to run analyze_experimental_results.py on the good runs
with the correct paths for David's system.
"""

import sys
import os
import json
import pandas as pd
from pathlib import Path

# Add the scripts directory to the path
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

# Import the analyzer class
from analyze_experimental_results import ExperimentalResultsAnalyzer

def main():
    """Run analysis on the good runs."""
    
    # Define the good run timestamps
    good_runs = [
        "20250914T110424Z",  # llama gsm8k
        "20250914T033113Z",  # llama superglue
        "20250913T105258Z",  # llama toolqa
        "20250914T045203Z",  # llama mathbench
        "20250914T071025Z",  # gpt-4 gsm8k
        "20250914T071025Z",  # gpt-4 superglue (same timestamp as gsm8k)
        "20250913T110235Z",  # gpt-4 toolqa
        "20250910T133420Z",  # gpt-4 mathbench
        "20250913T145036Z",  # gpt-4o gsm8k
        "20250913T121131Z",  # gpt-4o superglue
        "20250913T083152Z",  # gpt-4o toolqa
        "20250910T132536Z",  # gpt-4o mathbench
        "20250913T142510Z",  # gpt-4o-mini gsm8k
        "20250913T120434Z",  # gpt-4o-mini superglue
        "20250913T081723Z",  # gpt-4o-mini toolqa
        "20250910T131758Z",  # gpt-4o-mini mathbench
        "20250913T213057Z",  # claude-haiku gsm8k
        "20250913T122909Z",  # claude-haiku superglue
        "20250913T082455Z",  # claude-haiku toolqa
        "20250910T135124Z",  # claude-haiku mathbench
        "20250913T215757Z",  # claude-sonnet gsm8k
        "20250913T124632Z",  # claude-sonnet superglue
        "20250913T084811Z",  # claude-sonnet toolqa
        "20250910T142409Z",  # claude-sonnet mathbench
    ]
    
    print("🚀 Starting analysis of good experimental runs...")
    print(f"📁 Using runs directory: {Path.cwd() / 'runs'}")
    print(f"🎯 Analyzing {len(good_runs)} specific run timestamps")
    
    # Create analyzer with correct path
    runs_dir = Path.cwd() / "runs"
    analyzer = ExperimentalResultsAnalyzer(runs_dir=str(runs_dir))
    
    # Set output directory
    output_dir = Path("phase3_complete_analysis")
    output_dir.mkdir(exist_ok=True)
    analyzer.output_dir = output_dir
    
    # Filter to only analyze the good runs
    def filter_good_runs():
        """Override the analyze_local_runs method to only process good runs."""
        print("🔍 Filtering to good runs only...")
        
        good_run_dirs = []
        for run_dir in analyzer.runs_dir.iterdir():
            if run_dir.is_dir():
                run_name = run_dir.name
                # Extract timestamp from run name
                timestamp = run_name.split('__')[0]
                if timestamp in good_runs:
                    good_run_dirs.append(run_dir)
                    print(f"  ✅ Including: {run_name}")
                else:
                    print(f"  ⏭️  Skipping: {run_name}")
        
        print(f"📊 Found {len(good_run_dirs)} good runs to analyze")
        
        # Process only the good runs
        for run_dir in good_run_dirs:
            try:
                run_name = run_dir.name
                print(f"📋 Processing: {run_name}")
                
                # Extract experiment info
                exp_info = analyzer.extract_experiment_info(run_name)
                if not exp_info:
                    print(f"  ⚠️  Could not parse run name: {run_name}")
                    continue
                
                # Load metrics
                metrics_file = run_dir / "metrics.json"
                if not metrics_file.exists():
                    print(f"  ⚠️  No metrics.json found: {run_name}")
                    continue
                
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                
                # Add to results
                result_row = {
                    'run_name': run_name,
                    'run_dir': str(run_dir),
                    **exp_info,
                    **metrics
                }
                
                analyzer.results_df = pd.concat([analyzer.results_df, pd.DataFrame([result_row])], ignore_index=True)
                print(f"  ✅ Added to analysis")
                
            except Exception as e:
                print(f"  ❌ Error processing {run_name}: {e}")
                continue
    
    # Override the method
    analyzer.analyze_local_runs = filter_good_runs
    
    # Run the analysis
    try:
        results = analyzer.run_full_analysis()
        
        if results is not None:
            print(f"\n📋 Analysis Summary:")
            print(f"  • {len(results)} total experimental runs analyzed")
            print(f"  • Data standardized and saved to {output_dir}/")
            print(f"  • Ready for scaling analysis and visualization")
        else:
            print("❌ Analysis failed - no valid experimental data found")
            
    except Exception as e:
        print(f"❌ Analysis failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
