#!/usr/bin/env python3
"""
Scaling Experiment CLI

Runs self-correction experiments across multiple models and datasets
for the scaling study.
"""

import argparse
import os
import sys
import json
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from experiments.scaling_experiment_runner import ScalingExperimentRunner, ScalingExperimentConfig
from utils.multi_model_manager import MultiModelManager

def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    parser = argparse.ArgumentParser(description="Run scaling experiments")
    
    # Experiment configuration
    parser.add_argument("--phase", choices=["1", "2", "3"], default="1",
                       help="Experiment phase (1=validation, 2=medium, 3=full)")
    parser.add_argument("--models", nargs="+", 
                       help="Specific models to test (overrides phase defaults)")
    parser.add_argument("--datasets", nargs="+",
                       help="Specific datasets to test (overrides phase defaults)")
    parser.add_argument("--sample-size", type=int,
                       help="Sample size (overrides phase defaults)")
    parser.add_argument("--max-turns", type=int, default=3,
                       help="Maximum self-correction turns")
    
    # Output configuration
    parser.add_argument("--output-dir", default="outputs/scaling_experiments",
                       help="Output directory for results")
    
    # Model management
    parser.add_argument("--list-models", action="store_true",
                       help="List available models and their status")
    parser.add_argument("--estimate-costs", action="store_true",
                       help="Estimate costs for the experiment")
    
    # Logging
    parser.add_argument("--log-level", default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Initialize model manager
    model_manager = MultiModelManager()
    
    # List models if requested
    if args.list_models:
        print("\n=== Available Models ===")
        available_models = model_manager.get_available_models()
        for model in available_models:
            status = "✓ Available" if model.available else "✗ No API key"
            print(f"{model.name:15} | {model.size_category:8} | {model.provider:12} | {status}")
        return
    
    # Estimate costs if requested
    if args.estimate_costs:
        print("\n=== Cost Estimation ===")
        
        # Get phase configuration
        phase_configs = {
            "1": {"models": ["gpt-4o-mini", "claude-haiku"], "datasets": ["toolqa"], "sample_size": 100},
            "2": {"models": ["gpt-4o-mini", "claude-haiku", "gpt-4o", "claude-sonnet"], "datasets": ["toolqa", "superglue"], "sample_size": 500},
            "3": {"models": ["gpt-4o-mini", "claude-haiku", "gpt-4o", "claude-sonnet", "gpt-4", "claude-opus"], "datasets": ["toolqa", "superglue", "college_math", "humaneval"], "sample_size": 1000}
        }
        
        phase_config = phase_configs[args.phase]
        models = args.models or phase_config["models"]
        datasets = args.datasets or phase_config["datasets"]
        sample_size = args.sample_size or phase_config["sample_size"]
        
        cost_breakdown = model_manager.estimate_experiment_cost(
            models=models,
            datasets=datasets,
            sample_sizes=[sample_size]
        )
        
        print(f"Phase {args.phase} estimated cost: ${cost_breakdown['total_cost']:.2f}")
        print("\nBreakdown by model:")
        for model, cost in cost_breakdown['breakdown'].items():
            print(f"  {model:15} | ${cost:.2f}")
        return
    
    # Create experiment configuration
    phase_configs = {
        "1": {"models": ["gpt-4o-mini", "claude-haiku"], "datasets": ["toolqa"], "sample_size": 100},
        "2": {"models": ["gpt-4o-mini", "claude-haiku", "gpt-4o", "claude-sonnet"], "datasets": ["toolqa", "superglue"], "sample_size": 500},
        "3": {"models": ["gpt-4o-mini", "claude-haiku", "gpt-4o", "claude-sonnet", "gpt-4", "claude-opus"], "datasets": ["toolqa", "superglue", "college_math", "humaneval"], "sample_size": 1000}
    }
    
    phase_config = phase_configs[args.phase]
    models = args.models or phase_config["models"]
    datasets = args.datasets or phase_config["datasets"]
    sample_size = args.sample_size or phase_config["sample_size"]
    
    config = ScalingExperimentConfig(
        models=models,
        datasets=datasets,
        sample_sizes=[sample_size],
        max_turns=args.max_turns,
        output_dir=args.output_dir
    )
    
    # Run experiments
    print(f"\n=== Starting Phase {args.phase} Scaling Experiment ===")
    print(f"Models: {', '.join(models)}")
    print(f"Datasets: {', '.join(datasets)}")
    print(f"Sample size: {sample_size}")
    print(f"Max turns: {args.max_turns}")
    print(f"Output directory: {args.output_dir}")
    
    try:
        runner = ScalingExperimentRunner(config)
        results = runner.run_phase_experiments(f"phase{args.phase}")
        
        print(f"\n=== Phase {args.phase} Complete ===")
        print(f"Total experiments: {len(results)}")
        print(f"Total cost: ${sum(r.cost for r in results):.2f}")
        
        # Generate analysis report
        analysis = runner.generate_analysis_report()
        print(f"Analysis report saved to: {args.output_dir}/analysis_report.json")
        
        # Print summary
        print("\n=== Summary ===")
        for model in models:
            model_results = [r for r in results if r.model_name == model]
            if model_results:
                avg_improvement = sum(r.improvement for r in model_results) / len(model_results)
                total_cost = sum(r.cost for r in model_results)
                print(f"{model:15} | Avg improvement: {avg_improvement:.3f} | Cost: ${total_cost:.2f}")
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()