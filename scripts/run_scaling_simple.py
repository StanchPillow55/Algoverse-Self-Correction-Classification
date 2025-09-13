#!/usr/bin/env python3
"""
Simple Scaling Experiment Runner

Runs self-correction experiments across multiple models using your existing pipeline.
This integrates with your current infrastructure to minimize changes.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def run_model_experiment(model_name, dataset_path, output_dir, max_turns=3):
    """Run experiment for a single model using existing pipeline."""
    
    # Map model names to your existing provider system
    model_mapping = {
        "gpt-4o-mini": "openai",
        "gpt-4o": "openai", 
        "gpt-4": "openai",
        "claude-haiku": "anthropic",
        "claude-sonnet": "anthropic",
        "claude-opus": "anthropic",
        "llama-70b": "replicate"
    }
    
    # Map display names to actual API model names
    api_model_mapping = {
        "gpt-4o-mini": "gpt-4o-mini",
        "gpt-4o": "gpt-4o", 
        "gpt-4": "gpt-4",
        "claude-haiku": "claude-3-haiku-20240307",
        # Update Sonnet to current ID; keep an explicit 3.5 alias
        "claude-sonnet": "claude-3-5-sonnet-20241022",
        "claude-sonnet-3.5": "claude-3-5-sonnet-20241022",
        "claude-opus": "claude-3-opus-20240229",
        "llama-70b": "meta/meta-llama-3-70b"
    }
    
    provider = model_mapping.get(model_name, "openai")
    api_model_name = api_model_mapping.get(model_name, model_name)
    
    # Set environment variables for the model
    os.environ["PROVIDER"] = provider
    os.environ["DEMO_MODE"] = "0"  # Use real API
    
    # Create output file
    output_file = output_dir / f"{model_name}_scaling_result.json"
    
    # Run using your existing main.py
    cmd = f"""
    python -m src.main run \
        --dataset {dataset_path} \
        --out {output_file} \
        --max-turns {max_turns} \
        --provider {provider} \
        --model {api_model_name}
    """
    
    print(f"Running: {model_name} on {dataset_path}")
    print(f"Command: {cmd.strip()}")
    
    # Execute the command
    result = os.system(cmd)
    
    if result == 0:
        print(f"✓ {model_name} completed successfully")
        return True
    else:
        print(f"✗ {model_name} failed with exit code {result}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run simple scaling experiments")
    parser.add_argument("--models", nargs="+", 
                       default=["gpt-4o-mini", "claude-haiku", "gpt-4o"],
                       help="Models to test")
    parser.add_argument("--dataset", required=True,
                       help="Path to dataset CSV file")
    parser.add_argument("--output-dir", default="outputs/scaling_simple",
                       help="Output directory for results")
    parser.add_argument("--max-turns", type=int, default=3,
                       help="Maximum self-correction turns")
    parser.add_argument("--phase", choices=["1", "2", "3"], default="1",
                       help="Experiment phase (1=validation, 2=medium, 3=full)")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Phase-specific model selection
    if args.phase == "1":
        # Phase 1: Cheap validation
        models = ["gpt-4o-mini", "claude-haiku"]
        print("🚀 Phase 1: Validation (Cheap Models)")
    elif args.phase == "2":
        # Phase 2: Medium scale
        models = ["gpt-4o-mini", "claude-haiku", "gpt-4o", "claude-sonnet"]
        print("🚀 Phase 2: Medium Scale")
    else:
        # Phase 3: Full scale
        if args.models == ["gpt-4o-mini", "claude-haiku", "gpt-4o"]:  # Default models
            models = ["gpt-4o-mini", "claude-haiku", "gpt-4o", "claude-sonnet", "llama-70b", "gpt-4"]
        else:
            models = args.models
        print("🚀 Phase 3: Full Scale")
    
    print(f"Models: {', '.join(models)}")
    print(f"Dataset: {args.dataset}")
    print(f"Max turns: {args.max_turns}")
    print(f"Output: {output_dir}")
    print()
    
    # Check if dataset exists
    if not Path(args.dataset).exists():
        print(f"❌ Dataset not found: {args.dataset}")
        return 1
    
    # Run experiments
    results = {}
    start_time = time.time()
    
    for i, model in enumerate(models, 1):
        print(f"[{i}/{len(models)}] Running {model}...")
        
        success = run_model_experiment(
            model, 
            args.dataset, 
            output_dir, 
            args.max_turns
        )
        
        results[model] = success
        
        if success:
            print(f"✓ {model} completed")
        else:
            print(f"✗ {model} failed")
        
        print()
    
    # Summary
    total_time = time.time() - start_time
    successful = sum(results.values())
    total = len(results)
    
    print("=" * 50)
    print("EXPERIMENT SUMMARY")
    print("=" * 50)
    print(f"Total time: {total_time:.1f} seconds")
    print(f"Successful: {successful}/{total}")
    print()
    
    for model, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{model:15} | {status}")
    
    # Save summary
    summary = {
        "phase": args.phase,
        "models": models,
        "dataset": args.dataset,
        "max_turns": args.max_turns,
        "results": results,
        "total_time": total_time,
        "successful": successful,
        "total": total,
        "timestamp": time.time()
    }
    
    summary_file = output_dir / "experiment_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to: {summary_file}")
    
    if successful == total:
        print("🎉 All experiments completed successfully!")
        return 0
    else:
        print("⚠️  Some experiments failed. Check logs for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())