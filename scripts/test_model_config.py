#!/usr/bin/env python3
"""
Test Model Configuration System

Tests the model configuration system for the scaling study.
"""

import json
import os
from pathlib import Path

def test_model_config():
    """Test the model configuration system."""
    config_path = Path("configs/scaling_models.json")
    
    if not config_path.exists():
        print("❌ Model config file not found")
        return False
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print("📋 Model Configuration Test")
    print("=" * 30)
    
    # Test models
    models = config.get('models', [])
    print(f"Total models: {len(models)}")
    
    # Group by size category
    by_category = {}
    for model in models:
        category = model.get('size_category', 'unknown')
        if category not in by_category:
            by_category[category] = []
        by_category[category].append(model['name'])
    
    print("\nModels by category:")
    for category, model_list in by_category.items():
        print(f"  {category}: {', '.join(model_list)}")
    
    # Test datasets
    datasets = config.get('datasets', [])
    print(f"\nTotal datasets: {len(datasets)}")
    for dataset in datasets:
        print(f"  {dataset['name']}: {dataset['description']}")
    
    # Test experiment phases
    phases = config.get('experiment_phases', {})
    print(f"\nExperiment phases: {len(phases)}")
    for phase_name, phase_config in phases.items():
        print(f"  {phase_name}: {phase_config['description']}")
        print(f"    Models: {', '.join(phase_config['models'])}")
        print(f"    Datasets: {', '.join(phase_config['datasets'])}")
        print(f"    Sample size: {phase_config['sample_size']}")
    
    # Test API key availability
    print("\nAPI Key Availability:")
    api_keys = {
        "openai": os.getenv("OPENAI_API_KEY"),
        "anthropic": os.getenv("ANTHROPIC_API_KEY"),
        "replicate": os.getenv("REPLICATE_API_TOKEN")
    }
    
    for provider, key in api_keys.items():
        status = "✓ Available" if key else "✗ Missing"
        print(f"  {provider}: {status}")
    
    # Test model availability
    print("\nModel Availability:")
    for model in models:
        provider = model['provider']
        has_key = api_keys.get(provider) is not None
        available = model.get('available', True) and has_key
        status = "✓ Available" if available else "✗ Not available"
        print(f"  {model['name']:15} | {status}")
    
    return True

def test_cost_estimation():
    """Test cost estimation for different phases."""
    config_path = Path("configs/scaling_models.json")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print("\n💰 Cost Estimation Test")
    print("=" * 30)
    
    # Model costs
    models = config.get('models', [])
    model_costs = {model['name']: model['estimated_cost_per_1k_tokens'] for model in models}
    
    print("Model costs per 1k tokens:")
    for name, cost in model_costs.items():
        print(f"  {name:15} | ${cost:.4f}")
    
    # Phase cost estimation
    phases = config.get('experiment_phases', {})
    
    print("\nPhase cost estimation:")
    for phase_name, phase_config in phases.items():
        total_cost = 0
        models = phase_config['models']
        datasets = phase_config['datasets']
        sample_size = phase_config['sample_size']
        
        # Estimate tokens per sample (conservative)
        tokens_per_sample = 200
        turns_per_sample = 3
        
        for model_name in models:
            if model_name in model_costs:
                cost_per_1k = model_costs[model_name]
                total_tokens = sample_size * tokens_per_sample * turns_per_sample
                cost = (total_tokens / 1000) * cost_per_1k
                total_cost += cost
        
        print(f"  {phase_name:15} | ${total_cost:.2f}")

if __name__ == "__main__":
    print("🧪 Testing Model Configuration System")
    print("=" * 40)
    
    success = test_model_config()
    if success:
        test_cost_estimation()
        print("\n✅ Model configuration system working!")
    else:
        print("\n❌ Model configuration system failed!")
