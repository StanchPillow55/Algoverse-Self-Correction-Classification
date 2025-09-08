#!/usr/bin/env python3
"""
Scaling Study Cost Estimation

Estimates costs for the scaling study experiments across different phases.
"""

import json
import argparse
from pathlib import Path

def load_model_config():
    """Load model configuration from JSON file."""
    config_path = Path("configs/scaling_models.json")
    
    if not config_path.exists():
        print("❌ Model config file not found")
        return None
    
    with open(config_path, 'r') as f:
        return json.load(f)

def estimate_phase_costs(config):
    """Estimate costs for each phase."""
    phases = config.get('experiment_phases', {})
    
    print("💰 Scaling Study Cost Estimation")
    print("=" * 40)
    
    total_cost = 0.0
    
    for phase_name, phase_config in phases.items():
        models = phase_config['models']
        datasets = phase_config['datasets']
        sample_size = phase_config['sample_size']
        
        # Get model costs
        model_costs = {}
        for model_data in config.get('models', []):
            if model_data['name'] in models:
                model_costs[model_data['name']] = model_data['estimated_cost_per_1k_tokens']
        
        # Estimate tokens per sample (conservative)
        tokens_per_sample = 200  # Input + output tokens
        turns_per_sample = 3     # Average self-correction turns
        
        phase_cost = 0.0
        model_breakdown = {}
        
        for model_name in models:
            if model_name in model_costs:
                cost_per_1k = model_costs[model_name]
                total_tokens = sample_size * tokens_per_sample * turns_per_sample
                cost = (total_tokens / 1000) * cost_per_1k
                model_breakdown[model_name] = cost
                phase_cost += cost
        
        total_cost += phase_cost
        
        print(f"\n{phase_name.upper()}:")
        print(f"  Models: {', '.join(models)}")
        print(f"  Datasets: {', '.join(datasets)}")
        print(f"  Sample size: {sample_size}")
        print(f"  Estimated cost: ${phase_cost:.2f}")
        
        print("  Model breakdown:")
        for model, cost in model_breakdown.items():
            print(f"    {model:15} | ${cost:.2f}")
    
    print(f"\nTOTAL ESTIMATED COST: ${total_cost:.2f}")
    
    return total_cost

def estimate_custom_experiment(models, datasets, sample_sizes):
    """Estimate costs for a custom experiment."""
    config = load_model_config()
    if not config:
        return
    
    print(f"\n💰 Custom Experiment Cost Estimation")
    print("=" * 40)
    print(f"Models: {', '.join(models)}")
    print(f"Datasets: {', '.join(datasets)}")
    print(f"Sample sizes: {sample_sizes}")
    
    # Get model costs
    model_costs = {}
    for model_data in config.get('models', []):
        if model_data['name'] in models:
            model_costs[model_data['name']] = model_data['estimated_cost_per_1k_tokens']
    
    # Estimate tokens per sample
    tokens_per_sample = 200
    turns_per_sample = 3
    
    total_cost = 0.0
    breakdown = {}
    
    for model_name in models:
        if model_name in model_costs:
            model_cost = 0.0
            cost_per_1k = model_costs[model_name]
            
            for dataset in datasets:
                for sample_size in sample_sizes:
                    total_tokens = sample_size * tokens_per_sample * turns_per_sample
                    cost = (total_tokens / 1000) * cost_per_1k
                    model_cost += cost
            
            breakdown[model_name] = model_cost
            total_cost += model_cost
    
    print(f"\nEstimated cost: ${total_cost:.2f}")
    print("\nModel breakdown:")
    for model, cost in breakdown.items():
        print(f"  {model:15} | ${cost:.2f}")
    
    return total_cost

def main():
    parser = argparse.ArgumentParser(description="Estimate scaling study costs")
    parser.add_argument("--phase", choices=["1", "2", "3", "all"], default="all",
                       help="Phase to estimate (1=validation, 2=medium, 3=full, all=all phases)")
    parser.add_argument("--models", nargs="+",
                       help="Custom models to estimate")
    parser.add_argument("--datasets", nargs="+",
                       help="Custom datasets to estimate")
    parser.add_argument("--sample-sizes", nargs="+", type=int,
                       help="Custom sample sizes to estimate")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_model_config()
    if not config:
        return 1
    
    if args.phase == "all":
        # Estimate all phases
        estimate_phase_costs(config)
    elif args.models and args.datasets and args.sample_sizes:
        # Custom experiment
        estimate_custom_experiment(args.models, args.datasets, args.sample_sizes)
    else:
        # Single phase
        phases = config.get('experiment_phases', {})
        phase_name = f"phase{args.phase}"
        
        if phase_name in phases:
            phase_config = phases[phase_name]
            models = phase_config['models']
            datasets = phase_config['datasets']
            sample_size = phase_config['sample_size']
            
            estimate_custom_experiment(models, datasets, [sample_size])
        else:
            print(f"❌ Phase {args.phase} not found")
            return 1
    
    return 0

if __name__ == "__main__":
    exit(main())