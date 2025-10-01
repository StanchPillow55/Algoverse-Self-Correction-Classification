#!/usr/bin/env python3
"""
Cost estimation for remaining scaling study experiments.
Based on current results analysis and scaling_models.json configuration.
"""

import json
import pandas as pd
from pathlib import Path

def load_config():
    """Load model and dataset configuration"""
    config_path = Path("configs/scaling_models.json")
    with open(config_path) as f:
        return json.load(f)

def analyze_completed_experiments():
    """Analyze what experiments are already completed"""
    results_path = Path("comprehensive_analysis_output/comprehensive_standardized_results.csv")
    
    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        return pd.DataFrame()
    
    df = pd.read_csv(results_path)
    
    # Filter for valid, non-demo experiments with substantial sample sizes
    df = df[
        (df['valid'] == True) & 
        (df['model'] != 'dev') & 
        (df['n_samples'] >= 50)  # Meaningful experiments only
    ]
    
    return df

def estimate_costs_for_remaining():
    """Estimate costs for experiments still needed"""
    config = load_config()
    completed_df = analyze_completed_experiments()
    
    # Target experiments based on your plan
    target_models = ['gpt-4o-mini', 'claude-haiku', 'gpt-4o', 'claude-sonnet', 'gpt-4', 'claude-opus']
    target_datasets = ['gsm8k', 'humaneval', 'superglue', 'toolqa', 'mathbench']
    target_sample_sizes = {
        'gsm8k': 1000,
        'humaneval': 164,
        'superglue': 500, 
        'toolqa': 500,
        'mathbench': 500
    }
    
    # Model cost mapping from config
    model_costs = {}
    for model in config['models']:
        model_costs[model['name']] = model['estimated_cost_per_1k_tokens']
    
    # Estimate tokens per sample (conservative estimates)
    tokens_per_sample = {
        'gsm8k': 1000,      # Math reasoning with 3 turns
        'humaneval': 800,   # Code generation
        'superglue': 600,   # Language understanding
        'toolqa': 1200,     # Tool usage (complex)
        'mathbench': 1000   # Math problems
    }
    
    print("=== REMAINING EXPERIMENT COST ESTIMATION ===\n")
    
    total_cost = 0
    missing_experiments = []
    
    for dataset in target_datasets:
        print(f"\n📊 {dataset.upper()}")
        dataset_cost = 0
        
        for model in target_models:
            # Check if experiment already completed
            existing = completed_df[
                (completed_df['dataset_clean'].str.lower() == dataset.lower()) & 
                (completed_df['model_clean'].str.contains(model.split('-')[0], case=False, na=False)) &
                (completed_df['n_samples'] >= target_sample_sizes.get(dataset, 100))
            ]
            
            if len(existing) > 0:
                print(f"  ✅ {model}: COMPLETED ({existing.iloc[0]['n_samples']} samples, {existing.iloc[0]['accuracy_mean']:.3f} accuracy)")
                continue
            
            # Calculate cost for missing experiment
            if model in model_costs:
                cost_per_token = model_costs[model]
                sample_size = target_sample_sizes.get(dataset, 100)
                tokens_needed = tokens_per_sample[dataset] * sample_size
                experiment_cost = (tokens_needed / 1000) * cost_per_token
                
                dataset_cost += experiment_cost
                missing_experiments.append({
                    'dataset': dataset,
                    'model': model,
                    'samples': sample_size,
                    'cost': experiment_cost
                })
                
                print(f"  ❌ {model}: MISSING (est. ${experiment_cost:.2f} for {sample_size} samples)")
            else:
                print(f"  ⚠️  {model}: Cost data unavailable")
        
        print(f"  💰 Dataset subtotal: ${dataset_cost:.2f}")
        total_cost += dataset_cost
    
    print(f"\n=== SUMMARY ===")
    print(f"Total missing experiments: {len(missing_experiments)}")
    print(f"Estimated total cost: ${total_cost:.2f}")
    print(f"Budget remaining: ${500 - total_cost:.2f}" + (" ✅" if total_cost <= 500 else " ⚠️"))
    
    # Priority recommendations
    print(f"\n=== PRIORITY RECOMMENDATIONS ===")
    
    # Sort by cost-effectiveness (lower cost first)
    missing_experiments.sort(key=lambda x: x['cost'])
    
    running_cost = 0
    priority_experiments = []
    
    for exp in missing_experiments:
        if running_cost + exp['cost'] <= 500:
            priority_experiments.append(exp)
            running_cost += exp['cost']
        else:
            break
    
    print(f"Within $500 budget, you can complete {len(priority_experiments)} experiments:")
    for exp in priority_experiments[:10]:  # Show top 10
        print(f"  • {exp['dataset']}/{exp['model']}: {exp['samples']} samples (${exp['cost']:.2f})")
    
    if len(priority_experiments) < len(missing_experiments):
        print(f"\nRemaining {len(missing_experiments) - len(priority_experiments)} experiments would need additional ${total_cost - running_cost:.2f}")
    
    return missing_experiments, total_cost

def identify_ensemble_opportunities():
    """Identify datasets ready for ensemble experiments"""
    completed_df = analyze_completed_experiments()
    
    print(f"\n=== ENSEMBLE READINESS ===")
    
    for dataset in ['gsm8k', 'humaneval', 'superglue', 'toolqa', 'mathbench']:
        dataset_results = completed_df[
            completed_df['dataset_clean'].str.lower() == dataset.lower()
        ]
        
        unique_models = len(dataset_results['model_clean'].unique())
        if unique_models >= 3:
            print(f"  ✅ {dataset.upper()}: {unique_models} models ready for ensemble")
            print(f"     Models: {', '.join(dataset_results['model_clean'].unique())}")
        else:
            print(f"  ❌ {dataset.upper()}: Only {unique_models} models completed")

if __name__ == "__main__":
    try:
        missing_experiments, total_cost = estimate_costs_for_remaining()
        identify_ensemble_opportunities()
        
        # Generate actionable next steps
        print(f"\n=== IMMEDIATE ACTION PLAN ===")
        print("1. Focus on non-Llama models first (available now)")
        print("2. Prioritize ToolQA and College Math completion (identified gaps)")
        print("3. Run ensemble experiments on GSM8K and HumanEval (ready datasets)")
        print("4. Setup Together.ai integration while experiments run")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you're in the project root directory with config files available.")