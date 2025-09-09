#!/usr/bin/env python3
"""
Complete David's Tasks Runner

Runs all remaining David's tasks with proper trace formatting and validation.
"""

import sys
import json
import time
import subprocess
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.enhanced_trace_formatter import EnhancedTraceFormatter
from src.utils.scaling_analyzer import ScalingAnalyzer
from src.utils.result_aggregator import ResultAggregator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CompleteDavidTasksRunner:
    """Runs all of David's remaining tasks."""
    
    def __init__(self, output_dir: str = "outputs/complete_david_tasks"):
        """Initialize the complete runner."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.trace_formatter = EnhancedTraceFormatter(str(self.output_dir / "enhanced_traces"))
        self.scaling_analyzer = ScalingAnalyzer()
        self.result_aggregator = ResultAggregator()
        
        # Task configurations
        self.task_configs = {
            "6.1": {
                "name": "Complete 2-model validation",
                "models": ["gpt-4o-mini", "claude-haiku"],
                "datasets": ["gsm8k"],
                "sample_sizes": [100],
                "description": "Both providers with proper validation"
            },
            "6.2": {
                "name": "Medium scale experiments",
                "models": ["gpt-4o-mini", "claude-haiku", "gpt-4o", "claude-sonnet"],
                "datasets": ["gsm8k", "humaneval"],
                "sample_sizes": [100, 500],
                "description": "4 models × 2 datasets × 500 samples"
            },
            "6.3": {
                "name": "Full scale experiments",
                "models": ["gpt-4o-mini", "claude-haiku", "gpt-4o", "claude-sonnet", "llama-70b", "gpt-4", "claude-opus"],
                "datasets": ["gsm8k", "humaneval", "toolqa", "mathbench"],
                "sample_sizes": [100, 500, 1000],
                "description": "7 models × 5 datasets × 1000 samples"
            }
        }
    
    def run_task_6_1_complete_validation(self) -> Dict[str, Any]:
        """Complete Task 6.1: Finish 2-model validation (both providers)."""
        logger.info("🚀 Running Task 6.1: Complete 2-model validation")
        
        config = self.task_configs["6.1"]
        results = {
            "task": "6.1",
            "start_time": time.time(),
            "experiments": [],
            "formatted_traces": {}
        }
        
        # Run experiments for both models
        for model in config["models"]:
            for dataset in config["datasets"]:
                for sample_size in config["sample_sizes"]:
                    logger.info(f"   Running {model} on {dataset} ({sample_size} samples)")
                    
                    # Determine provider
                    provider = "openai" if "gpt" in model else "anthropic"
                    
                    # Create output directory
                    exp_dir = self.output_dir / f"task_6_1_{model}_{dataset}_{sample_size}"
                    exp_dir.mkdir(exist_ok=True)
                    
                    # Run experiment
                    traces_file = exp_dir / "traces.jsonl"
                    cmd = [
                        "python", "src/main.py", "run",
                        "--dataset", f"data/{dataset}/test_{sample_size}.jsonl",
                        "--out", str(traces_file),
                        "--max-turns", "3",
                        "--provider", provider,
                        "--model", model
                    ]
                    
                    try:
                        # Set PYTHONPATH for subprocess
                        env = os.environ.copy()
                        env['PYTHONPATH'] = str(project_root)
                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
                        
                        if result.returncode == 0:
                            logger.info(f"   ✅ {model} completed successfully")
                            
                            # Format traces
                            if traces_file.exists():
                                formatted = self.trace_formatter.format_experiment_traces(
                                    str(traces_file), f"task_6_1_{model}_{dataset}"
                                )
                                results["formatted_traces"][f"{model}_{dataset}"] = formatted
                                
                                # Calculate metrics
                                with open(traces_file, 'r') as f:
                                    data = json.load(f)
                                    traces = data.get('traces', [])
                                
                                total_samples = len(traces)
                                correct_samples = sum(1 for trace in traces if trace.get('final_accuracy', trace.get('final_correct', 0)) == 1)
                                accuracy = correct_samples / total_samples if total_samples > 0 else 0
                                
                                results["experiments"].append({
                                    "model": model,
                                    "dataset": dataset,
                                    "sample_size": sample_size,
                                    "accuracy": accuracy,
                                    "correct_samples": correct_samples,
                                    "total_samples": total_samples,
                                    "traces_file": str(traces_file)
                                })
                        else:
                            logger.error(f"   ❌ {model} failed: {result.stderr}")
                            
                    except subprocess.TimeoutExpired:
                        logger.error(f"   ❌ {model} timed out")
                    except Exception as e:
                        logger.error(f"   ❌ {model} error: {e}")
        
        results["end_time"] = time.time()
        results["duration"] = results["end_time"] - results["start_time"]
        
        # Save results
        results_file = self.output_dir / "task_6_1_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"✅ Task 6.1 completed in {results['duration']:.1f}s")
        return results
    
    def run_task_6_2_medium_scale(self) -> Dict[str, Any]:
        """Run Task 6.2: Medium scale experiments."""
        logger.info("🚀 Running Task 6.2: Medium scale experiments")
        
        config = self.task_configs["6.2"]
        results = {
            "task": "6.2",
            "start_time": time.time(),
            "experiments": [],
            "formatted_traces": {}
        }
        
        # Run experiments
        for model in config["models"]:
            for dataset in config["datasets"]:
                for sample_size in config["sample_sizes"]:
                    logger.info(f"   Running {model} on {dataset} ({sample_size} samples)")
                    
                    # Determine provider
                    if "gpt" in model:
                        provider = "openai"
                    elif "claude" in model:
                        provider = "anthropic"
                    else:
                        provider = "openai"  # Default
                    
                    # Create output directory
                    exp_dir = self.output_dir / f"task_6_2_{model}_{dataset}_{sample_size}"
                    exp_dir.mkdir(exist_ok=True)
                    
                    # Run experiment
                    traces_file = exp_dir / "traces.jsonl"
                    cmd = [
                        "python", "src/main.py", "run",
                        "--dataset", f"data/{dataset}/test_{sample_size}.jsonl",
                        "--out", str(traces_file),
                        "--max-turns", "3",
                        "--provider", provider,
                        "--model", model
                    ]
                    
                    try:
                        # Set PYTHONPATH for subprocess
                        env = os.environ.copy()
                        env['PYTHONPATH'] = str(project_root)
                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
                        
                        if result.returncode == 0:
                            logger.info(f"   ✅ {model} on {dataset} completed")
                            
                            # Format traces
                            if traces_file.exists():
                                formatted = self.trace_formatter.format_experiment_traces(
                                    str(traces_file), f"task_6_2_{model}_{dataset}_{sample_size}"
                                )
                                results["formatted_traces"][f"{model}_{dataset}_{sample_size}"] = formatted
                                
                                # Calculate metrics
                                with open(traces_file, 'r') as f:
                                    data = json.load(f)
                                    traces = data.get('traces', [])
                                
                                total_samples = len(traces)
                                correct_samples = sum(1 for trace in traces if trace.get('final_accuracy', trace.get('final_correct', 0)) == 1)
                                accuracy = correct_samples / total_samples if total_samples > 0 else 0
                                
                                results["experiments"].append({
                                    "model": model,
                                    "dataset": dataset,
                                    "sample_size": sample_size,
                                    "accuracy": accuracy,
                                    "correct_samples": correct_samples,
                                    "total_samples": total_samples,
                                    "traces_file": str(traces_file)
                                })
                        else:
                            logger.error(f"   ❌ {model} on {dataset} failed: {result.stderr}")
                            
                    except subprocess.TimeoutExpired:
                        logger.error(f"   ❌ {model} on {dataset} timed out")
                    except Exception as e:
                        logger.error(f"   ❌ {model} on {dataset} error: {e}")
        
        results["end_time"] = time.time()
        results["duration"] = results["end_time"] - results["start_time"]
        
        # Save results
        results_file = self.output_dir / "task_6_2_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"✅ Task 6.2 completed in {results['duration']:.1f}s")
        return results
    
    def run_task_7_1_power_law_analysis(self, experiment_results: List[Dict]) -> Dict[str, Any]:
        """Run Task 7.1: Power-law scaling exponents analysis."""
        logger.info("🚀 Running Task 7.1: Power-law scaling exponents analysis")
        
        # Prepare data for power-law fitting
        model_data = {}
        for result in experiment_results:
            model = result.get('model', 'unknown')
            if model not in model_data:
                model_data[model] = []
            
            model_data[model].append({
                'accuracy': result.get('accuracy', 0),
                'sample_size': result.get('sample_size', 0),
                'dataset': result.get('dataset', 'unknown')
            })
        
        # Fit power laws
        power_law_results = {}
        for model, data in model_data.items():
            if len(data) >= 3:  # Need at least 3 points for power law
                accuracies = [d['accuracy'] for d in data]
                sample_sizes = [d['sample_size'] for d in data]
                
                # Simple power law fit: accuracy = a * sample_size^b
                try:
                    # Use log-linear regression
                    log_acc = np.log(accuracies + 1e-10)  # Add small value to avoid log(0)
                    log_size = np.log(sample_sizes)
                    
                    # Linear regression: log(accuracy) = log(a) + b * log(sample_size)
                    coeffs = np.polyfit(log_size, log_acc, 1)
                    b = coeffs[0]  # Power law exponent
                    log_a = coeffs[1]
                    a = np.exp(log_a)
                    
                    # Calculate R²
                    predicted = a * np.power(sample_sizes, b)
                    ss_res = np.sum((accuracies - predicted) ** 2)
                    ss_tot = np.sum((accuracies - np.mean(accuracies)) ** 2)
                    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                    
                    power_law_results[model] = {
                        'exponent': b,
                        'coefficient': a,
                        'r_squared': r_squared,
                        'data_points': len(data),
                        'valid_fit': r_squared > 0.85
                    }
                except Exception as e:
                    power_law_results[model] = {
                        'error': str(e),
                        'data_points': len(data)
                    }
        
        # Save results
        results_file = self.output_dir / "task_7_1_power_law_results.json"
        with open(results_file, 'w') as f:
            json.dump(power_law_results, f, indent=2)
        
        logger.info(f"✅ Task 7.1 completed - Power law analysis saved to {results_file}")
        return power_law_results
    
    def run_task_8_1_confidence_intervals(self, experiment_results: List[Dict]) -> Dict[str, Any]:
        """Run Task 8.1: 95% confidence intervals for all metrics."""
        logger.info("🚀 Running Task 8.1: 95% confidence intervals")
        
        ci_results = {}
        
        # Group by model
        model_groups = {}
        for result in experiment_results:
            model = result.get('model', 'unknown')
            if model not in model_groups:
                model_groups[model] = []
            model_groups[model].append(result)
        
        for model, results in model_groups.items():
            accuracies = [r.get('accuracy', 0) for r in results]
            sample_sizes = [r.get('total_samples', 0) for r in results]
            
            if accuracies:
                # Calculate 95% CI for accuracy
                mean_acc = np.mean(accuracies)
                std_acc = np.std(accuracies)
                n = len(accuracies)
                
                # Standard error
                se = std_acc / np.sqrt(n) if n > 0 else 0
                
                # 95% CI using t-distribution (approximate with normal for large n)
                margin_error = 1.96 * se
                ci_lower = max(0, mean_acc - margin_error)
                ci_upper = min(1, mean_acc + margin_error)
                
                ci_results[model] = {
                    'mean_accuracy': mean_acc,
                    'std_accuracy': std_acc,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'margin_error': margin_error,
                    'sample_count': n,
                    'total_samples': sum(sample_sizes)
                }
        
        # Save results
        results_file = self.output_dir / "task_8_1_confidence_intervals.json"
        with open(results_file, 'w') as f:
            json.dump(ci_results, f, indent=2)
        
        logger.info(f"✅ Task 8.1 completed - Confidence intervals saved to {results_file}")
        return ci_results
    
    def run_all_tasks(self) -> Dict[str, Any]:
        """Run all of David's remaining tasks."""
        logger.info("🎯 Running All David's Remaining Tasks")
        logger.info("=" * 60)
        
        all_results = {
            "start_time": time.time(),
            "tasks": {},
            "summary": {}
        }
        
        # Task 6.1: Complete 2-model validation
        try:
            task_6_1_results = self.run_task_6_1_complete_validation()
            all_results["tasks"]["6.1"] = task_6_1_results
            logger.info("✅ Task 6.1 completed")
        except Exception as e:
            logger.error(f"❌ Task 6.1 failed: {e}")
            all_results["tasks"]["6.1"] = {"error": str(e)}
        
        # Task 6.2: Medium scale experiments
        try:
            task_6_2_results = self.run_task_6_2_medium_scale()
            all_results["tasks"]["6.2"] = task_6_2_results
            logger.info("✅ Task 6.2 completed")
        except Exception as e:
            logger.error(f"❌ Task 6.2 failed: {e}")
            all_results["tasks"]["6.2"] = {"error": str(e)}
        
        # Collect all experiment results for analysis
        all_experiments = []
        for task_name, task_data in all_results["tasks"].items():
            if "experiments" in task_data:
                all_experiments.extend(task_data["experiments"])
        
        # Task 7.1: Power-law analysis
        if all_experiments:
            try:
                task_7_1_results = self.run_task_7_1_power_law_analysis(all_experiments)
                all_results["tasks"]["7.1"] = task_7_1_results
                logger.info("✅ Task 7.1 completed")
            except Exception as e:
                logger.error(f"❌ Task 7.1 failed: {e}")
                all_results["tasks"]["7.1"] = {"error": str(e)}
        
        # Task 8.1: Confidence intervals
        if all_experiments:
            try:
                task_8_1_results = self.run_task_8_1_confidence_intervals(all_experiments)
                all_results["tasks"]["8.1"] = task_8_1_results
                logger.info("✅ Task 8.1 completed")
            except Exception as e:
                logger.error(f"❌ Task 8.1 failed: {e}")
                all_results["tasks"]["8.1"] = {"error": str(e)}
        
        # Generate summary
        all_results["end_time"] = time.time()
        all_results["duration"] = all_results["end_time"] - all_results["start_time"]
        
        all_results["summary"] = {
            "total_duration": all_results["duration"],
            "tasks_completed": len([k for k, v in all_results["tasks"].items() if "error" not in v]),
            "tasks_failed": len([k for k, v in all_results["tasks"].items() if "error" in v]),
            "total_experiments": len(all_experiments),
            "formatted_traces": self._count_formatted_traces(all_results)
        }
        
        # Save all results
        results_file = self.output_dir / "all_david_tasks_results.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        logger.info(f"🎉 All David's tasks completed in {all_results['duration']:.1f}s")
        logger.info(f"📊 Results saved to: {results_file}")
        
        return all_results
    
    def _count_formatted_traces(self, results: Dict[str, Any]) -> Dict[str, int]:
        """Count formatted traces across all tasks."""
        trace_counts = {}
        for task_name, task_data in results["tasks"].items():
            if "formatted_traces" in task_data:
                trace_counts[task_name] = len(task_data["formatted_traces"])
        return trace_counts

def main():
    """Main function to run all David's tasks."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run all of David's remaining tasks")
    parser.add_argument("--task", choices=["6.1", "6.2", "7.1", "8.1", "all"], 
                       default="all", help="Which task to run")
    parser.add_argument("--output-dir", default="outputs/complete_david_tasks",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = CompleteDavidTasksRunner(args.output_dir)
    
    if args.task == "6.1":
        runner.run_task_6_1_complete_validation()
    elif args.task == "6.2":
        runner.run_task_6_2_medium_scale()
    elif args.task == "7.1":
        # Need some experiment results for this
        print("Task 7.1 requires experiment results. Run tasks 6.1 and 6.2 first.")
    elif args.task == "8.1":
        # Need some experiment results for this
        print("Task 8.1 requires experiment results. Run tasks 6.1 and 6.2 first.")
    elif args.task == "all":
        runner.run_all_tasks()

if __name__ == "__main__":
    main()
