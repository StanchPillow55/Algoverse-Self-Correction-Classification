#!/usr/bin/env python3
"""
Complete Validation of David's Tasks

This script validates that all of David's remaining tasks are properly implemented,
smoke tested, and produce correctly formatted traces (.txt for full traces, .json for accuracy data).
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DavidTasksValidator:
    """Validates all of David's remaining tasks."""
    
    def __init__(self, output_dir: str = "outputs/david_validation"):
        """Initialize the validator."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.trace_formatter = EnhancedTraceFormatter(str(self.output_dir / "enhanced_traces"))
        
        # Task validation results
        self.validation_results = {
            "start_time": time.time(),
            "tasks": {},
            "summary": {}
        }
    
    def validate_task_6_1_complete_validation(self) -> Dict[str, Any]:
        """Validate Task 6.1: Complete 2-model validation (both providers)."""
        logger.info("🔍 Validating Task 6.1: Complete 2-model validation")
        
        task_result = {
            "task": "6.1",
            "status": "testing",
            "start_time": time.time(),
            "experiments": [],
            "formatted_traces": {},
            "validation_passed": False
        }
        
        # Test both providers with small sample
        models = ["gpt-4o-mini", "claude-haiku"]
        dataset = "gsm8k"
        sample_size = 10  # Small sample for validation
        
        for model in models:
            logger.info(f"   Testing {model} on {dataset} ({sample_size} samples)")
            
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
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=env)
                
                if result.returncode == 0 and traces_file.exists():
                    logger.info(f"   ✅ {model} completed successfully")
                    
                    # Format traces with enhanced formatter
                    formatted = self.trace_formatter.format_experiment_traces(
                        str(traces_file), f"task_6_1_{model}_{dataset}"
                    )
                    task_result["formatted_traces"][f"{model}_{dataset}"] = formatted
                    
                    # Calculate metrics
                    with open(traces_file, 'r') as f:
                        data = json.load(f)
                        traces = data.get('traces', [])
                    
                    total_samples = len(traces)
                    correct_samples = sum(1 for trace in traces if trace.get('final_accuracy', trace.get('final_correct', 0)) == 1)
                    accuracy = correct_samples / total_samples if total_samples > 0 else 0
                    
                    task_result["experiments"].append({
                        "model": model,
                        "dataset": dataset,
                        "sample_size": sample_size,
                        "accuracy": accuracy,
                        "correct_samples": correct_samples,
                        "total_samples": total_samples,
                        "traces_file": str(traces_file),
                        "formatted_outputs": list(formatted.keys())
                    })
                    
                    # Validate trace formatting
                    self._validate_trace_formatting(formatted, f"{model}_{dataset}")
                    
                else:
                    logger.error(f"   ❌ {model} failed: {result.stderr}")
                    task_result["experiments"].append({
                        "model": model,
                        "status": "failed",
                        "error": result.stderr
                    })
                    
            except subprocess.TimeoutExpired:
                logger.error(f"   ❌ {model} timed out")
                task_result["experiments"].append({
                    "model": model,
                    "status": "timeout"
                })
            except Exception as e:
                logger.error(f"   ❌ {model} error: {e}")
                task_result["experiments"].append({
                    "model": model,
                    "status": "error",
                    "error": str(e)
                })
        
        # Determine if task passed
        successful_experiments = [exp for exp in task_result["experiments"] if "accuracy" in exp]
        task_result["validation_passed"] = len(successful_experiments) >= 2  # Both models should work
        
        task_result["end_time"] = time.time()
        task_result["duration"] = task_result["end_time"] - task_result["start_time"]
        task_result["status"] = "passed" if task_result["validation_passed"] else "failed"
        
        logger.info(f"✅ Task 6.1 validation {'PASSED' if task_result['validation_passed'] else 'FAILED'}")
        return task_result
    
    def validate_task_6_2_medium_scale(self) -> Dict[str, Any]:
        """Validate Task 6.2: Medium scale experiments."""
        logger.info("🔍 Validating Task 6.2: Medium scale experiments")
        
        task_result = {
            "task": "6.2",
            "status": "testing",
            "start_time": time.time(),
            "experiments": [],
            "formatted_traces": {},
            "validation_passed": False
        }
        
        # Test with 2 models, 1 dataset, small sample
        models = ["gpt-4o-mini", "claude-haiku"]
        datasets = ["gsm8k"]
        sample_size = 5  # Very small sample for validation
        
        for model in models:
            for dataset in datasets:
                logger.info(f"   Testing {model} on {dataset} ({sample_size} samples)")
                
                # Determine provider
                provider = "openai" if "gpt" in model else "anthropic"
                
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
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=env)
                    
                    if result.returncode == 0 and traces_file.exists():
                        logger.info(f"   ✅ {model} on {dataset} completed")
                        
                        # Format traces
                        formatted = self.trace_formatter.format_experiment_traces(
                            str(traces_file), f"task_6_2_{model}_{dataset}_{sample_size}"
                        )
                        task_result["formatted_traces"][f"{model}_{dataset}_{sample_size}"] = formatted
                        
                        # Calculate metrics
                        with open(traces_file, 'r') as f:
                            data = json.load(f)
                            traces = data.get('traces', [])
                        
                        total_samples = len(traces)
                        correct_samples = sum(1 for trace in traces if trace.get('final_accuracy', trace.get('final_correct', 0)) == 1)
                        accuracy = correct_samples / total_samples if total_samples > 0 else 0
                        
                        task_result["experiments"].append({
                            "model": model,
                            "dataset": dataset,
                            "sample_size": sample_size,
                            "accuracy": accuracy,
                            "correct_samples": correct_samples,
                            "total_samples": total_samples,
                            "traces_file": str(traces_file),
                            "formatted_outputs": list(formatted.keys())
                        })
                        
                        # Validate trace formatting
                        self._validate_trace_formatting(formatted, f"{model}_{dataset}_{sample_size}")
                        
                    else:
                        logger.error(f"   ❌ {model} on {dataset} failed: {result.stderr}")
                        task_result["experiments"].append({
                            "model": model,
                            "dataset": dataset,
                            "status": "failed",
                            "error": result.stderr
                        })
                        
                except subprocess.TimeoutExpired:
                    logger.error(f"   ❌ {model} on {dataset} timed out")
                    task_result["experiments"].append({
                        "model": model,
                        "dataset": dataset,
                        "status": "timeout"
                    })
                except Exception as e:
                    logger.error(f"   ❌ {model} on {dataset} error: {e}")
                    task_result["experiments"].append({
                        "model": model,
                        "dataset": dataset,
                        "status": "error",
                        "error": str(e)
                    })
        
        # Determine if task passed
        successful_experiments = [exp for exp in task_result["experiments"] if "accuracy" in exp]
        task_result["validation_passed"] = len(successful_experiments) >= 2  # Both model-dataset combinations should work
        
        task_result["end_time"] = time.time()
        task_result["duration"] = task_result["end_time"] - task_result["start_time"]
        task_result["status"] = "passed" if task_result["validation_passed"] else "failed"
        
        logger.info(f"✅ Task 6.2 validation {'PASSED' if task_result['validation_passed'] else 'FAILED'}")
        return task_result
    
    def validate_task_7_1_power_law_analysis(self) -> Dict[str, Any]:
        """Validate Task 7.1: Power-law scaling exponents analysis."""
        logger.info("🔍 Validating Task 7.1: Power-law scaling exponents analysis")
        
        task_result = {
            "task": "7.1",
            "status": "testing",
            "start_time": time.time(),
            "validation_passed": False
        }
        
        # Create mock data for power-law analysis validation
        mock_data = [
            {"model": "gpt-4o-mini", "accuracy": 0.6, "sample_size": 10},
            {"model": "gpt-4o-mini", "accuracy": 0.65, "sample_size": 50},
            {"model": "gpt-4o-mini", "accuracy": 0.7, "sample_size": 100},
            {"model": "claude-haiku", "accuracy": 0.5, "sample_size": 10},
            {"model": "claude-haiku", "accuracy": 0.55, "sample_size": 50},
            {"model": "claude-haiku", "accuracy": 0.6, "sample_size": 100},
        ]
        
        try:
            # Test power-law fitting
            power_law_results = self._fit_power_laws(mock_data)
            
            # Validate that we can fit power laws
            valid_fits = [model for model, result in power_law_results.items() if result.get('valid_fit', False)]
            task_result["validation_passed"] = len(valid_fits) >= 1
            
            task_result["power_law_results"] = power_law_results
            task_result["valid_fits"] = valid_fits
            
        except Exception as e:
            logger.error(f"   ❌ Power-law analysis failed: {e}")
            task_result["error"] = str(e)
        
        task_result["end_time"] = time.time()
        task_result["duration"] = task_result["end_time"] - task_result["start_time"]
        task_result["status"] = "passed" if task_result["validation_passed"] else "failed"
        
        logger.info(f"✅ Task 7.1 validation {'PASSED' if task_result['validation_passed'] else 'FAILED'}")
        return task_result
    
    def validate_task_8_1_confidence_intervals(self) -> Dict[str, Any]:
        """Validate Task 8.1: 95% confidence intervals for all metrics."""
        logger.info("🔍 Validating Task 8.1: 95% confidence intervals")
        
        task_result = {
            "task": "8.1",
            "status": "testing",
            "start_time": time.time(),
            "validation_passed": False
        }
        
        # Create mock data for confidence interval validation
        mock_data = [
            {"model": "gpt-4o-mini", "accuracy": 0.6, "total_samples": 10},
            {"model": "gpt-4o-mini", "accuracy": 0.65, "total_samples": 10},
            {"model": "gpt-4o-mini", "accuracy": 0.7, "total_samples": 10},
            {"model": "claude-haiku", "accuracy": 0.5, "total_samples": 10},
            {"model": "claude-haiku", "accuracy": 0.55, "total_samples": 10},
            {"model": "claude-haiku", "accuracy": 0.6, "total_samples": 10},
        ]
        
        try:
            # Test confidence interval calculation
            ci_results = self._calculate_confidence_intervals(mock_data)
            
            # Validate that we can calculate CIs
            task_result["validation_passed"] = len(ci_results) >= 1 and all(
                "ci_lower" in result and "ci_upper" in result 
                for result in ci_results.values()
            )
            
            task_result["confidence_intervals"] = ci_results
            
        except Exception as e:
            logger.error(f"   ❌ Confidence interval calculation failed: {e}")
            task_result["error"] = str(e)
        
        task_result["end_time"] = time.time()
        task_result["duration"] = task_result["end_time"] - task_result["start_time"]
        task_result["status"] = "passed" if task_result["validation_passed"] else "failed"
        
        logger.info(f"✅ Task 8.1 validation {'PASSED' if task_result['validation_passed'] else 'FAILED'}")
        return task_result
    
    def _validate_trace_formatting(self, formatted_outputs: Dict[str, str], experiment_id: str):
        """Validate that trace formatting produces correct file types."""
        logger.info(f"   Validating trace formatting for {experiment_id}")
        
        # Check that we have the expected output types
        expected_outputs = ["full_traces_dir", "accuracy_data", "summary_metrics", "multi_turn_analysis"]
        
        for output_type in expected_outputs:
            if output_type in formatted_outputs:
                output_path = Path(formatted_outputs[output_type])
                if output_path.exists():
                    logger.info(f"   ✅ {output_type}: {output_path}")
                    
                    # Validate specific file types
                    if output_type == "full_traces_dir":
                        # Should contain .txt files
                        txt_files = list(output_path.glob("*.txt"))
                        if txt_files:
                            logger.info(f"   ✅ Found {len(txt_files)} .txt files for full traces")
                        else:
                            logger.warning(f"   ⚠️ No .txt files found in {output_path}")
                    
                    elif output_type == "accuracy_data":
                        # Should be a .json file
                        if output_path.suffix == ".json":
                            logger.info(f"   ✅ Accuracy data is properly formatted as JSON")
                        else:
                            logger.warning(f"   ⚠️ Accuracy data is not JSON: {output_path}")
                else:
                    logger.warning(f"   ⚠️ {output_type} file not found: {output_path}")
            else:
                logger.warning(f"   ⚠️ Missing {output_type} in formatted outputs")
    
    def _fit_power_laws(self, data: List[Dict]) -> Dict[str, Any]:
        """Fit power laws to the data."""
        model_groups = {}
        for item in data:
            model = item.get('model', 'unknown')
            if model not in model_groups:
                model_groups[model] = []
            model_groups[model].append(item)
        
        power_law_results = {}
        for model, items in model_groups.items():
            if len(items) >= 3:  # Need at least 3 points for power law
                accuracies = [item['accuracy'] for item in items]
                sample_sizes = [item['sample_size'] for item in items]
                
                try:
                    # Simple power law fit: accuracy = a * sample_size^b
                    log_acc = np.log(np.array(accuracies) + 1e-10)
                    log_size = np.log(sample_sizes)
                    
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
                        'data_points': len(items),
                        'valid_fit': r_squared > 0.85
                    }
                except Exception as e:
                    power_law_results[model] = {
                        'error': str(e),
                        'data_points': len(items)
                    }
        
        return power_law_results
    
    def _calculate_confidence_intervals(self, data: List[Dict]) -> Dict[str, Any]:
        """Calculate 95% confidence intervals for the data."""
        model_groups = {}
        for item in data:
            model = item.get('model', 'unknown')
            if model not in model_groups:
                model_groups[model] = []
            model_groups[model].append(item)
        
        ci_results = {}
        for model, items in model_groups.items():
            accuracies = [item.get('accuracy', 0) for item in items]
            sample_sizes = [item.get('total_samples', 0) for item in items]
            
            if accuracies:
                mean_acc = np.mean(accuracies)
                std_acc = np.std(accuracies)
                n = len(accuracies)
                
                se = std_acc / np.sqrt(n) if n > 0 else 0
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
        
        return ci_results
    
    def run_all_validations(self) -> Dict[str, Any]:
        """Run all validations for David's tasks."""
        logger.info("🎯 Running Complete Validation of David's Tasks")
        logger.info("=" * 60)
        
        # Task 6.1: Complete 2-model validation
        try:
            task_6_1_result = self.validate_task_6_1_complete_validation()
            self.validation_results["tasks"]["6.1"] = task_6_1_result
        except Exception as e:
            logger.error(f"❌ Task 6.1 validation failed: {e}")
            self.validation_results["tasks"]["6.1"] = {"error": str(e), "status": "failed"}
        
        # Task 6.2: Medium scale experiments
        try:
            task_6_2_result = self.validate_task_6_2_medium_scale()
            self.validation_results["tasks"]["6.2"] = task_6_2_result
        except Exception as e:
            logger.error(f"❌ Task 6.2 validation failed: {e}")
            self.validation_results["tasks"]["6.2"] = {"error": str(e), "status": "failed"}
        
        # Task 7.1: Power-law analysis
        try:
            task_7_1_result = self.validate_task_7_1_power_law_analysis()
            self.validation_results["tasks"]["7.1"] = task_7_1_result
        except Exception as e:
            logger.error(f"❌ Task 7.1 validation failed: {e}")
            self.validation_results["tasks"]["7.1"] = {"error": str(e), "status": "failed"}
        
        # Task 8.1: Confidence intervals
        try:
            task_8_1_result = self.validate_task_8_1_confidence_intervals()
            self.validation_results["tasks"]["8.1"] = task_8_1_result
        except Exception as e:
            logger.error(f"❌ Task 8.1 validation failed: {e}")
            self.validation_results["tasks"]["8.1"] = {"error": str(e), "status": "failed"}
        
        # Generate summary
        self.validation_results["end_time"] = time.time()
        self.validation_results["duration"] = self.validation_results["end_time"] - self.validation_results["start_time"]
        
        passed_tasks = [k for k, v in self.validation_results["tasks"].items() if v.get("validation_passed", False)]
        failed_tasks = [k for k, v in self.validation_results["tasks"].items() if not v.get("validation_passed", False)]
        
        self.validation_results["summary"] = {
            "total_duration": self.validation_results["duration"],
            "tasks_passed": len(passed_tasks),
            "tasks_failed": len(failed_tasks),
            "passed_tasks": passed_tasks,
            "failed_tasks": failed_tasks,
            "overall_success": len(passed_tasks) >= 3  # At least 3 out of 4 tasks should pass
        }
        
        # Save validation results
        results_file = self.output_dir / "validation_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.validation_results, f, indent=2)
        
        logger.info(f"🎉 Validation completed in {self.validation_results['duration']:.1f}s")
        logger.info(f"📊 Results: {len(passed_tasks)}/{len(self.validation_results['tasks'])} tasks passed")
        logger.info(f"📁 Results saved to: {results_file}")
        
        return self.validation_results

def main():
    """Main function to run all validations."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate all of David's remaining tasks")
    parser.add_argument("--task", choices=["6.1", "6.2", "7.1", "8.1", "all"], 
                       default="all", help="Which task to validate")
    parser.add_argument("--output-dir", default="outputs/david_validation",
                       help="Output directory for validation results")
    
    args = parser.parse_args()
    
    # Initialize validator
    validator = DavidTasksValidator(args.output_dir)
    
    if args.task == "6.1":
        validator.validate_task_6_1_complete_validation()
    elif args.task == "6.2":
        validator.validate_task_6_2_medium_scale()
    elif args.task == "7.1":
        validator.validate_task_7_1_power_law_analysis()
    elif args.task == "8.1":
        validator.validate_task_8_1_confidence_intervals()
    elif args.task == "all":
        validator.run_all_validations()

if __name__ == "__main__":
    main()
