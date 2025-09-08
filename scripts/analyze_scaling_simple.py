#!/usr/bin/env python3
"""
Simple Scaling Analysis

Analyzes scaling laws in self-correction experiments without complex imports.
"""

import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

def power_law_fit(x_data, y_data):
    """Fit power law: y = a * x^b using basic linear regression on log-transformed data."""
    if len(x_data) < 2:
        return {"exponent": 0, "coefficient": 0, "r_squared": 0, "equation": "y = 0"}
    
    # Remove zeros and negative values for log fitting
    mask = (x_data > 0) & (y_data > 0)
    x_clean = x_data[mask]
    y_clean = y_data[mask]
    
    if len(x_clean) < 2:
        return {"exponent": 0, "coefficient": 0, "r_squared": 0, "equation": "y = 0"}
    
    try:
        # Linear regression on log-transformed data
        log_x = np.log(x_clean)
        log_y = np.log(y_clean)
        
        # Linear regression: log(y) = log(a) + b * log(x)
        n = len(log_x)
        sum_log_x = np.sum(log_x)
        sum_log_y = np.sum(log_y)
        sum_log_x_sq = np.sum(log_x ** 2)
        sum_log_xy = np.sum(log_x * log_y)
        
        # Calculate coefficients
        b = (n * sum_log_xy - sum_log_x * sum_log_y) / (n * sum_log_x_sq - sum_log_x ** 2)
        log_a = (sum_log_y - b * sum_log_x) / n
        a = np.exp(log_a)
        
        # Calculate R-squared
        y_pred = a * np.power(x_clean, b)
        ss_res = np.sum((y_clean - y_pred) ** 2)
        ss_tot = np.sum((y_clean - np.mean(y_clean)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        equation = f"y = {a:.4f} * x^{b:.4f}"
        
        return {
            "exponent": b,
            "coefficient": a,
            "r_squared": r_squared,
            "equation": equation
        }
        
    except Exception as e:
        print(f"Power law fitting failed: {e}")
        return {"exponent": 0, "coefficient": 0, "r_squared": 0, "equation": "y = 0"}

def load_experiment_results(results_dir):
    """Load experiment results for scaling analysis."""
    results_dir = Path(results_dir)
    results = []
    
    # Model size mapping (approximate parameter counts)
    model_sizes = {
        "gpt-4o-mini": 1.8,  # 1.8B parameters
        "claude-haiku": 3.0,  # 3B parameters
        "gpt-4o": 8.0,  # 8B parameters
        "claude-sonnet": 70.0,  # 70B parameters
        "llama-70b": 70.0,  # 70B parameters
        "gpt-4": 100.0,  # 100B+ parameters
        "claude-opus": 100.0,  # 100B+ parameters
    }
    
    # Model cost mapping
    model_costs = {
        "gpt-4o-mini": 0.00015,
        "claude-haiku": 0.00025,
        "gpt-4o": 0.0025,
        "claude-sonnet": 0.003,
        "llama-70b": 0.0007,
        "gpt-4": 0.03,
        "claude-opus": 0.015
    }
    
    # Find all result files
    result_files = list(results_dir.glob("*.json"))
    
    for file_path in result_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Extract model name from filename
            filename = file_path.stem
            parts = filename.split('_')
            model_name = parts[0] if parts else "unknown"
            
            # Get model size and cost
            model_size = model_sizes.get(model_name, 1.0)
            cost_per_1k = model_costs.get(model_name, 0.001)
            
            # Calculate metrics from traces
            traces = data.get('traces', [])
            if not traces:
                continue
            
            # Calculate improvement
            initial_correct = sum(1 for t in traces if t.get('turns', [{}])[0].get('accuracy', 0) == 1)
            final_correct = sum(1 for t in traces if t.get('final_accuracy', 0) == 1)
            
            initial_accuracy = initial_correct / len(traces)
            final_accuracy = final_correct / len(traces)
            improvement = final_accuracy - initial_accuracy
            
            # Estimate cost
            total_tokens = len(traces) * 200  # Rough estimate
            cost = (total_tokens / 1000) * cost_per_1k
            cost_efficiency = improvement / cost if cost > 0 else 0
            
            # Extract dataset name
            dataset_name = parts[1] if len(parts) > 1 else "unknown"
            
            results.append({
                'model_size': model_size,
                'improvement': improvement,
                'cost': cost,
                'cost_efficiency': cost_efficiency,
                'model_name': model_name,
                'dataset_name': dataset_name,
                'samples': len(traces)
            })
            
        except Exception as e:
            print(f"⚠️  Failed to load {file_path}: {e}")
            continue
    
    return results

def analyze_scaling(results):
    """Analyze scaling patterns in results."""
    if not results:
        return {"error": "No results to analyze"}
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    analysis = {
        "summary": {
            "total_experiments": len(results),
            "models_tested": df['model_name'].nunique(),
            "datasets_tested": df['dataset_name'].nunique(),
            "avg_improvement": df['improvement'].mean(),
            "max_improvement": df['improvement'].max(),
            "min_improvement": df['improvement'].min(),
            "total_cost": df['cost'].sum()
        },
        "scaling_laws": {},
        "recommendations": []
    }
    
    # Analyze scaling by model size
    if len(df) > 1:
        x_data = df['model_size'].values
        y_data = df['improvement'].values
        
        # Fit power law
        power_law_fit_result = power_law_fit(x_data, y_data)
        
        analysis["scaling_laws"]["model_size_vs_improvement"] = power_law_fit_result
        
        # Add interpretation
        exponent = power_law_fit_result["exponent"]
        if exponent > 0.5:
            interpretation = "Strong positive scaling - improvement increases rapidly with model size"
        elif exponent > 0.2:
            interpretation = "Moderate positive scaling - improvement increases with model size"
        elif exponent > 0:
            interpretation = "Weak positive scaling - slight improvement with model size"
        elif exponent > -0.2:
            interpretation = "Weak negative scaling - slight decrease with model size"
        else:
            interpretation = "Strong negative scaling - improvement decreases with model size"
        
        analysis["scaling_laws"]["model_size_vs_improvement"]["interpretation"] = interpretation
    
    # Analyze cost scaling
    if len(df) > 1:
        x_data = df['model_size'].values
        y_data = df['cost_efficiency'].values
        
        cost_fit = power_law_fit(x_data, y_data)
        analysis["scaling_laws"]["model_size_vs_cost_efficiency"] = cost_fit
    
    # Generate recommendations
    recommendations = []
    
    # Scaling recommendations
    if "model_size_vs_improvement" in analysis["scaling_laws"]:
        scaling = analysis["scaling_laws"]["model_size_vs_improvement"]
        exponent = scaling["exponent"]
        r_squared = scaling["r_squared"]
        
        if r_squared > 0.8:
            recommendations.append(f"Strong scaling law detected (R² = {r_squared:.3f})")
            if exponent > 0.3:
                recommendations.append("Self-correction benefits significantly from larger models")
            elif exponent > 0.1:
                recommendations.append("Self-correction benefits moderately from larger models")
            else:
                recommendations.append("Self-correction benefits minimally from larger models")
        else:
            recommendations.append(f"Weak scaling law (R² = {r_squared:.3f}) - model size may not be the primary factor")
    
    # Cost recommendations
    if len(df) > 0:
        most_efficient = df.loc[df['cost_efficiency'].idxmax(), 'model_name']
        recommendations.append(f"Most cost-efficient model: {most_efficient}")
    
    # General recommendations
    avg_improvement = analysis["summary"]["avg_improvement"]
    if avg_improvement > 0.1:
        recommendations.append("Self-correction shows significant improvement (>10%)")
    elif avg_improvement > 0.05:
        recommendations.append("Self-correction shows moderate improvement (5-10%)")
    else:
        recommendations.append("Self-correction shows minimal improvement (<5%)")
    
    analysis["recommendations"] = recommendations
    
    return analysis

def main():
    parser = argparse.ArgumentParser(description="Analyze scaling laws in experiments")
    parser.add_argument("--results-dir", default="outputs/scaling_experiments",
                       help="Directory containing experiment results")
    parser.add_argument("--output", default="outputs/scaling_analysis.json",
                       help="Output file for analysis results")
    
    args = parser.parse_args()
    
    print("🔬 Analyzing Scaling Laws")
    print("=" * 30)
    
    # Load results
    print(f"Loading results from: {args.results_dir}")
    results = load_experiment_results(args.results_dir)
    
    if not results:
        print("❌ No results found to analyze")
        return 1
    
    print(f"✓ Loaded {len(results)} experiment results")
    
    # Analyze scaling
    print("Analyzing scaling patterns...")
    analysis = analyze_scaling(results)
    
    # Print summary
    print("\n📊 Scaling Analysis Summary")
    print("=" * 40)
    
    summary = analysis.get("summary", {})
    print(f"Total experiments: {summary.get('total_experiments', 0)}")
    print(f"Models tested: {summary.get('models_tested', 0)}")
    print(f"Datasets tested: {summary.get('datasets_tested', 0)}")
    print(f"Average improvement: {summary.get('avg_improvement', 0):.3f}")
    print(f"Max improvement: {summary.get('max_improvement', 0):.3f}")
    print(f"Min improvement: {summary.get('min_improvement', 0):.3f}")
    print(f"Total cost: ${summary.get('total_cost', 0):.2f}")
    
    # Print scaling laws
    scaling_laws = analysis.get("scaling_laws", {})
    if scaling_laws:
        print("\n🔬 Scaling Laws")
        print("-" * 20)
        
        for law_name, law_data in scaling_laws.items():
            print(f"\n{law_name}:")
            print(f"  Equation: {law_data['equation']}")
            print(f"  R² = {law_data['r_squared']:.3f}")
            if 'interpretation' in law_data:
                print(f"  Interpretation: {law_data['interpretation']}")
    
    # Print recommendations
    recommendations = analysis.get("recommendations", [])
    if recommendations:
        print("\n💡 Recommendations")
        print("-" * 20)
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
    
    # Save analysis
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
    
    print(f"\n✅ Analysis saved to: {output_path}")
    
    return 0

if __name__ == "__main__":
    exit(main())
