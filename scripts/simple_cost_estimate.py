#!/usr/bin/env python3
"""
Simple Cost Estimation for Scaling Study
"""

def estimate_costs():
    """Estimate costs for the scaling study."""
    
    # Model configurations
    models = {
        "gpt-4o-mini": {"cost_per_1k": 0.00015, "category": "small"},
        "claude-haiku": {"cost_per_1k": 0.00025, "category": "small"},
        "gpt-4o": {"cost_per_1k": 0.0025, "category": "medium"},
        "claude-sonnet": {"cost_per_1k": 0.003, "category": "medium"},
        "llama-70b": {"cost_per_1k": 0.0007, "category": "medium"},
        "gpt-4": {"cost_per_1k": 0.03, "category": "large"},
        "claude-opus": {"cost_per_1k": 0.015, "category": "large"}
    }
    
    # Dataset configurations
    datasets = ["toolqa", "superglue", "mathbench", "gsm8k", "humaneval"]
    sample_sizes = [100, 500, 1000]
    
    # Token estimates
    tokens_per_sample = 200  # Input + output tokens per sample
    turns_per_sample = 3     # Average self-correction turns
    
    print("💰 Scaling Study Cost Estimation")
    print("=" * 40)
    print(f"Models: {len(models)}")
    print(f"Datasets: {len(datasets)}")
    print(f"Sample sizes: {sample_sizes}")
    print(f"Tokens per sample: {tokens_per_sample}")
    print(f"Turns per sample: {turns_per_sample}")
    print()
    
    total_cost = 0.0
    total_experiments = 0
    
    print("Cost breakdown by model:")
    print("-" * 50)
    
    for model_name, config in models.items():
        model_cost = 0.0
        experiments = 0
        
        for dataset in datasets:
            for sample_size in sample_sizes:
                # Calculate tokens for this experiment
                total_tokens = sample_size * tokens_per_sample * turns_per_sample
                cost = (total_tokens / 1000) * config["cost_per_1k"]
                
                model_cost += cost
                experiments += 1
                total_experiments += 1
        
        total_cost += model_cost
        cost_per_exp = model_cost / experiments if experiments > 0 else 0
        
        print(f"{model_name:15} | ${model_cost:8.2f} | ${cost_per_exp:.2f}/exp | {experiments:2d} exps | {config['category']}")
    
    print()
    print(f"Total experiments: {total_experiments}")
    print(f"Total estimated cost: ${total_cost:.2f}")
    print()
    
    # Phase recommendations
    print("Phased approach recommendations:")
    print("-" * 35)
    
    # Phase 1: Cheap validation
    phase1_models = ["gpt-4o-mini", "claude-haiku"]
    phase1_cost = 0
    for model_name in phase1_models:
        if model_name in models:
            for dataset in datasets:
                for sample_size in [100]:  # Small samples only
                    total_tokens = sample_size * tokens_per_sample * turns_per_sample
                    cost = (total_tokens / 1000) * models[model_name]["cost_per_1k"]
                    phase1_cost += cost
    
    print(f"Phase 1 (validation): ${phase1_cost:.2f}")
    print(f"  Models: {', '.join(phase1_models)}")
    print(f"  Sample size: 100")
    print(f"  Purpose: Validate approach before full investment")
    
    # Phase 2: Medium scale
    phase2_models = ["gpt-4o-mini", "claude-haiku", "gpt-4o", "claude-sonnet"]
    phase2_cost = 0
    for model_name in phase2_models:
        if model_name in models:
            for dataset in datasets:
                for sample_size in [100, 500]:
                    total_tokens = sample_size * tokens_per_sample * turns_per_sample
                    cost = (total_tokens / 1000) * models[model_name]["cost_per_1k"]
                    phase2_cost += cost
    
    print(f"Phase 2 (medium): ${phase2_cost:.2f}")
    print(f"  Models: {', '.join(phase2_models)}")
    print(f"  Sample sizes: 100, 500")
    print(f"  Purpose: Test scaling hypothesis with medium models")
    
    # Phase 3: Full scale
    print(f"Phase 3 (full): ${total_cost:.2f}")
    print(f"  All models and sample sizes")
    print(f"  Purpose: Complete scaling law analysis")
    
    print()
    print("Budget recommendations:")
    print("-" * 25)
    
    if total_cost > 1000:
        print("⚠️  High cost detected. Consider:")
        print("   - Starting with Phase 1 only")
        print("   - Reducing sample sizes")
        print("   - Focusing on fewer models initially")
    elif total_cost > 500:
        print("💡 Moderate cost. Good balance of coverage and budget.")
        print("   - Start with Phase 1, then Phase 2")
        print("   - Full scale only if results are promising")
    else:
        print("✅ Low cost. Consider adding more models or larger samples.")
    
    print()
    print("Expected ROI:")
    print("-" * 15)
    print("• 6 models × 5 datasets × 3 sample sizes = 90 data points")
    print("• Clear scaling law insights for practitioners")
    print("• Strong ICLR submission with actionable recommendations")
    print("• Reusable infrastructure for future experiments")
    
    return total_cost

if __name__ == "__main__":
    estimate_costs()
