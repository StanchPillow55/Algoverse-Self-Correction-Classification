#!/usr/bin/env python3
"""
Full Scaling Study Runner

Runs the complete scaling study across all phases and generates comprehensive analysis.
"""

import os
import sys
import time
import argparse
from pathlib import Path

def run_phase(phase, output_dir):
    """Run a specific phase of the scaling study."""
    print(f"\n🚀 Running Phase {phase}")
    print("=" * 40)
    
    if phase == "1":
        # Phase 1: Validation (2 models, 1 dataset, 100 samples)
        cmd = f"""
        python scripts/run_scaling_simple.py \
            --dataset data/scaling/toolqa_sample_100.csv \
            --phase 1 \
            --output-dir {output_dir}/phase1
        """
    elif phase == "2":
        # Phase 2: Medium scale (4 models, 2 datasets, 500 samples)
        cmd = f"""
        python scripts/run_scaling_simple.py \
            --dataset data/scaling/superglue_sample_500.csv \
            --phase 2 \
            --output-dir {output_dir}/phase2
        """
    elif phase == "3":
        # Phase 3: Full scale (6 models, 4 datasets, 1000 samples)
        cmd = f"""
        python scripts/run_scaling_simple.py \
            --dataset data/scaling/mathbench_sample_1000.csv \
            --phase 3 \
            --output-dir {output_dir}/phase3
        """
    else:
        print(f"❌ Unknown phase: {phase}")
        return False
    
    print(f"Command: {cmd.strip()}")
    
    # Execute the command
    result = os.system(cmd)
    
    if result == 0:
        print(f"✅ Phase {phase} completed successfully")
        return True
    else:
        print(f"❌ Phase {phase} failed with exit code {result}")
        return False

def analyze_results(output_dir):
    """Analyze results from all phases."""
    print(f"\n📊 Analyzing Results")
    print("=" * 40)
    
    # Analyze each phase
    for phase in ["1", "2", "3"]:
        phase_dir = f"{output_dir}/phase{phase}"
        if Path(phase_dir).exists():
            print(f"\nAnalyzing Phase {phase}...")
            
            # Run simple analysis
            cmd = f"""
            python scripts/analyze_results_simple.py \
                --results-dir {phase_dir} \
                --output {phase_dir}/analysis.json
            """
            
            result = os.system(cmd)
            if result == 0:
                print(f"✅ Phase {phase} analysis complete")
            else:
                print(f"⚠️  Phase {phase} analysis failed")
    
    # Run scaling analysis
    print(f"\nRunning scaling law analysis...")
    cmd = f"""
    python scripts/analyze_scaling_simple.py \
        --results-dir {output_dir} \
        --output {output_dir}/scaling_analysis.json
    """
    
    result = os.system(cmd)
    if result == 0:
        print("✅ Scaling analysis complete")
    else:
        print("⚠️  Scaling analysis failed")

def main():
    parser = argparse.ArgumentParser(description="Run full scaling study")
    parser.add_argument("--phases", nargs="+", default=["1", "2", "3"],
                       help="Phases to run (1=validation, 2=medium, 3=full)")
    parser.add_argument("--output-dir", default="outputs/full_scaling_study",
                       help="Output directory for all results")
    parser.add_argument("--skip-experiments", action="store_true",
                       help="Skip experiments and only run analysis")
    parser.add_argument("--analyze-only", action="store_true",
                       help="Only run analysis on existing results")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🔬 Full Scaling Study Runner")
    print("=" * 50)
    print(f"Phases: {', '.join(args.phases)}")
    print(f"Output: {output_dir}")
    print(f"Skip experiments: {args.skip_experiments}")
    print(f"Analyze only: {args.analyze_only}")
    
    # Check if API keys are available
    if not args.analyze_only and not args.skip_experiments:
        if not os.getenv("OPENAI_API_KEY") or not os.getenv("ANTHROPIC_API_KEY"):
            print("❌ API keys not found. Please set OPENAI_API_KEY and ANTHROPIC_API_KEY")
            print("   You can source your .env file: source .env")
            return 1
    
    start_time = time.time()
    
    # Run experiments
    if not args.analyze_only:
        success_count = 0
        total_phases = len(args.phases)
        
        for phase in args.phases:
            if run_phase(phase, str(output_dir)):
                success_count += 1
        
        print(f"\n📊 Experiment Summary")
        print("=" * 30)
        print(f"Successful phases: {success_count}/{total_phases}")
        
        if success_count == 0:
            print("❌ No phases completed successfully")
            return 1
        elif success_count < total_phases:
            print("⚠️  Some phases failed, but continuing with analysis")
    
    # Analyze results
    if not args.skip_experiments or args.analyze_only:
        analyze_results(str(output_dir))
    
    # Generate final report
    total_time = time.time() - start_time
    print(f"\n🎉 Scaling Study Complete!")
    print("=" * 40)
    print(f"Total time: {total_time:.1f} seconds")
    print(f"Results saved to: {output_dir}")
    
    # Print next steps
    print(f"\n📋 Next Steps:")
    print(f"1. Review results in: {output_dir}")
    print(f"2. Check analysis files for insights")
    print(f"3. Generate paper from findings")
    print(f"4. Submit to ICLR")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
