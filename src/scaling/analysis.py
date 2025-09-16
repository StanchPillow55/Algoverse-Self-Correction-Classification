#!/usr/bin/env python3
"""
Scaling Laws Analysis Pipeline

Tools for computing delta improvement, fitting power laws, and generating
scaling law statistics for the self-correction study.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
from typing import Dict, List, Tuple, Optional, Any
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass

try:
    from .model_registry import get_model_config, MODEL_REGISTRY
except ImportError:
    # Fallback for standalone execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))
    from model_registry import get_model_config, MODEL_REGISTRY


@dataclass
class ScalingResult:
    """Results from a scaling law analysis."""
    model_name: str
    parameter_count_b: float
    task_type: str
    initial_accuracy: float
    final_accuracy: float
    delta_improvement: float
    cost_per_sample: float
    improvement_per_dollar: float
    num_samples: int
    confidence_interval: Tuple[float, float]


@dataclass
class PowerLawFit:
    """Power law fit results."""
    scaling_exponent: float  # α in Δ ∝ ModelSize^α
    coefficient: float       # A in Δ = A × ModelSize^α
    r_squared: float
    confidence_interval: Tuple[float, float]
    p_value: float


def power_law_func(x: np.ndarray, a: float, alpha: float) -> np.ndarray:
    """Power law function: y = a * x^alpha"""
    return a * np.power(x, alpha)


def compute_delta_improvement(
    traces: List[Dict[str, Any]]
) -> Dict[str, float]:
    """
    Compute delta improvement from trace data.
    
    Args:
        traces: List of experiment traces
        
    Returns:
        Dictionary with accuracy metrics
    """
    if not traces:
        return {"initial_accuracy": 0.0, "final_accuracy": 0.0, "delta_improvement": 0.0}
    
    initial_accuracies = []
    final_accuracies = []
    
    for trace in traces:
        turns = trace.get('turns', [])
        if not turns:
            continue
            
        # Initial accuracy (first turn)
        initial_acc = turns[0].get('accuracy', 0)
        initial_accuracies.append(initial_acc)
        
        # Final accuracy (last turn)
        final_acc = turns[-1].get('accuracy', 0)
        final_accuracies.append(final_acc)
    
    if not initial_accuracies:
        return {"initial_accuracy": 0.0, "final_accuracy": 0.0, "delta_improvement": 0.0}
    
    initial_mean = np.mean(initial_accuracies)
    final_mean = np.mean(final_accuracies)
    delta = final_mean - initial_mean
    
    # Compute confidence intervals
    initial_ci = stats.t.interval(0.95, len(initial_accuracies)-1, 
                                 loc=initial_mean, 
                                 scale=stats.sem(initial_accuracies))
    final_ci = stats.t.interval(0.95, len(final_accuracies)-1,
                               loc=final_mean,
                               scale=stats.sem(final_accuracies))
    
    return {
        "initial_accuracy": initial_mean,
        "final_accuracy": final_mean,
        "delta_improvement": delta,
        "initial_ci": initial_ci,
        "final_ci": final_ci,
        "num_samples": len(traces)
    }


def fit_power_law(
    parameter_counts: List[float],
    improvements: List[float],
    improvement_errors: Optional[List[float]] = None
) -> PowerLawFit:
    """
    Fit power law to scaling data: Δ = A × ModelSize^α
    
    Args:
        parameter_counts: Model parameter counts in billions
        improvements: Delta improvements
        improvement_errors: Error bars for improvements
        
    Returns:
        PowerLawFit with scaling results
    """
    if len(parameter_counts) != len(improvements):
        raise ValueError("Parameter counts and improvements must have same length")
    
    # Convert to numpy arrays
    x = np.array(parameter_counts)
    y = np.array(improvements)
    
    # Filter out negative or zero values for log-space fitting
    valid_mask = (x > 0) & (y > 0)
    if not valid_mask.any():
        # Fallback for cases with no positive improvements
        return PowerLawFit(0.0, 0.0, 0.0, (0.0, 0.0), 1.0)
    
    x_valid = x[valid_mask]
    y_valid = y[valid_mask]
    
    if len(x_valid) < 3:
        # Need at least 3 points for meaningful fit
        return PowerLawFit(0.0, 0.0, 0.0, (0.0, 0.0), 1.0)
    
    try:
        # Fit power law using curve_fit
        initial_guess = [0.05, 0.3]  # A=0.05, α=0.3
        popt, pcov = curve_fit(power_law_func, x_valid, y_valid, 
                              p0=initial_guess, maxfev=5000)
        
        a_fit, alpha_fit = popt
        
        # Compute R-squared
        y_pred = power_law_func(x_valid, a_fit, alpha_fit)
        ss_res = np.sum((y_valid - y_pred) ** 2)
        ss_tot = np.sum((y_valid - np.mean(y_valid)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        # Compute confidence interval for alpha
        alpha_err = np.sqrt(pcov[1, 1]) if pcov[1, 1] > 0 else 0.0
        alpha_ci = (alpha_fit - 1.96 * alpha_err, alpha_fit + 1.96 * alpha_err)
        
        # Compute p-value (simplified)
        t_stat = alpha_fit / alpha_err if alpha_err > 0 else 0
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(x_valid) - 2))
        
        return PowerLawFit(
            scaling_exponent=alpha_fit,
            coefficient=a_fit,
            r_squared=r_squared,
            confidence_interval=alpha_ci,
            p_value=p_value
        )
        
    except Exception as e:
        print(f"Power law fitting failed: {e}")
        return PowerLawFit(0.0, 0.0, 0.0, (0.0, 0.0), 1.0)


def analyze_scaling_results(
    results_dir: Path,
    task_types: List[str] = None
) -> Dict[str, Any]:
    """
    Analyze scaling results from experiment outputs.
    
    Args:
        results_dir: Directory containing experiment results
        task_types: Task types to analyze (default: all)
        
    Returns:
        Dictionary with scaling analysis results
    """
    if task_types is None:
        task_types = ["toolqa", "superglue", "college_math", "humaneval"]
    
    scaling_results = []
    
    # Collect results from all experiments
    for result_file in results_dir.glob("**/traces.json"):
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
            
            # Extract model info from path or config
            config_file = result_file.parent / "config.json"
            model_name = "unknown"
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)
                model_name = config.get('model', 'unknown')
            
            # Get model configuration
            model_config = get_model_config(model_name)
            if not model_config:
                continue
                
            # Compute metrics
            traces = data.get('traces', [])
            metrics = compute_delta_improvement(traces)
            
            # Estimate cost
            cost_per_sample = model_config.cost_per_1k_tokens * 2  # Rough estimate
            improvement_per_dollar = (metrics['delta_improvement'] / cost_per_sample 
                                    if cost_per_sample > 0 else 0)
            
            # Determine task type from traces or config
            task_type = "unknown"
            if traces:
                first_qid = traces[0].get('qid', '')
                if 'humaneval' in first_qid.lower():
                    task_type = "humaneval"
                elif 'toolqa' in first_qid.lower():
                    task_type = "toolqa"
                elif any(task in first_qid.lower() for task in ['boolq', 'copa', 'rte']):
                    task_type = "superglue"
                else:
                    task_type = "college_math"
            
            result = ScalingResult(
                model_name=model_config.name,
                parameter_count_b=model_config.parameter_count_b,
                task_type=task_type,
                initial_accuracy=metrics['initial_accuracy'],
                final_accuracy=metrics['final_accuracy'],
                delta_improvement=metrics['delta_improvement'],
                cost_per_sample=cost_per_sample,
                improvement_per_dollar=improvement_per_dollar,
                num_samples=metrics['num_samples'],
                confidence_interval=metrics.get('final_ci', (0.0, 0.0))
            )
            
            scaling_results.append(result)
            
        except Exception as e:
            print(f"Error processing {result_file}: {e}")
            continue
    
    # Analyze scaling laws by task type
    analysis = {
        "scaling_results": scaling_results,
        "power_law_fits": {},
        "cost_benefit_analysis": {},
        "summary_stats": {}
    }
    
    for task_type in task_types:
        task_results = [r for r in scaling_results if r.task_type == task_type]
        if not task_results:
            continue
            
        # Extract data for power law fitting
        param_counts = [r.parameter_count_b for r in task_results]
        improvements = [r.delta_improvement for r in task_results]
        
        # Fit power law
        power_law = fit_power_law(param_counts, improvements)
        analysis["power_law_fits"][task_type] = power_law
        
        # Cost-benefit analysis
        cost_benefits = [r.improvement_per_dollar for r in task_results]
        analysis["cost_benefit_analysis"][task_type] = {
            "mean_improvement_per_dollar": np.mean(cost_benefits),
            "std_improvement_per_dollar": np.std(cost_benefits),
            "best_model": max(task_results, key=lambda x: x.improvement_per_dollar).model_name
        }
        
        # Summary statistics
        analysis["summary_stats"][task_type] = {
            "num_models": len(task_results),
            "mean_delta": np.mean(improvements),
            "std_delta": np.std(improvements),
            "min_params": min(param_counts),
            "max_params": max(param_counts)
        }
    
    return analysis


def generate_summary_tables(analysis: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """Generate summary tables for publication."""
    
    # Table 1: Results by Model Size Category
    scaling_results = analysis["scaling_results"]
    
    size_categories = {}
    for result in scaling_results:
        model_config = get_model_config(result.model_name.lower().replace(' ', '-'))
        if model_config:
            category = model_config.size_category
            if category not in size_categories:
                size_categories[category] = []
            size_categories[category].append(result)
    
    table1_data = []
    for category in ["Small", "Medium", "Large"]:
        if category in size_categories:
            results = size_categories[category]
            avg_delta = np.mean([r.delta_improvement for r in results])
            std_delta = np.std([r.delta_improvement for r in results])
            avg_cost = np.mean([r.cost_per_sample for r in results])
            avg_cost_benefit = np.mean([r.improvement_per_dollar for r in results])
            
            table1_data.append({
                "Size Category": category,
                "Models": len(results),
                "Avg Δ": f"{avg_delta:.3f} ± {std_delta:.3f}",
                "Cost per Sample": f"${avg_cost:.4f}",
                "Cost-Benefit Ratio": f"{avg_cost_benefit:.0f}"
            })
    
    table1 = pd.DataFrame(table1_data)
    
    # Table 2: Task-Specific Scaling Patterns
    table2_data = []
    for task_type, power_law in analysis["power_law_fits"].items():
        summary_stats = analysis["summary_stats"][task_type]
        
        table2_data.append({
            "Task Type": task_type,
            "Scaling Exponent": f"{power_law.scaling_exponent:.2f}",
            "R²": f"{power_law.r_squared:.3f}",
            "Best Model Size": f"{summary_stats['max_params']:.0f}B+",
            "Notes": get_task_notes(task_type, power_law.scaling_exponent)
        })
    
    table2 = pd.DataFrame(table2_data)
    
    return {"table1": table1, "table2": table2}


def get_task_notes(task_type: str, scaling_exponent: float) -> str:
    """Get notes for task scaling patterns."""
    if scaling_exponent > 0.32:
        strength = "strongest"
    elif scaling_exponent > 0.25:
        strength = "strong"
    else:
        strength = "weaker"
    
    notes_map = {
        "toolqa": f"External reasoning shows {strength} scaling",
        "superglue": f"Language understanding shows {strength} scaling", 
        "college_math": f"Mathematical reasoning shows {strength} scaling",
        "humaneval": f"Code generation shows {strength} scaling"
    }
    
    return notes_map.get(task_type, f"{strength} scaling pattern")


if __name__ == "__main__":
    # Test power law fitting
    param_counts = [1.8, 3.0, 8.0, 70.0, 175.0]
    improvements = [0.05, 0.08, 0.12, 0.18, 0.22]
    
    power_law = fit_power_law(param_counts, improvements)
    print(f"Scaling exponent: {power_law.scaling_exponent:.3f}")
    print(f"R-squared: {power_law.r_squared:.3f}")
    print(f"Formula: Δ = {power_law.coefficient:.3f} × ModelSize^{power_law.scaling_exponent:.3f}")