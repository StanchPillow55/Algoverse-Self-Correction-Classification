#!/usr/bin/env python3
"""
Scaling Study Experiment Runner

Orchestrates experiments across all 7 models and 4 datasets for the scaling laws study.
Runs with proper statistical sampling, cost tracking, and reproducibility controls.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import yaml

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.scaling.model_registry import MODEL_REGISTRY, get_model_config, estimate_experiment_cost


class ScalingStudyRunner:
    """Orchestrates the complete scaling study experiment."""
    
    def __init__(self, output_dir: str = "scaling_study_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Experimental settings
        self.temperature = 0.0
        self.max_turns = 3 
        self.num_runs = 3  # Statistical sampling
        self.sample_size = 100  # Per dataset
        
        # Dataset configurations
        self.datasets = {
            "toolqa": "data/scaling/toolqa_sample.csv",
            "superglue": "data/scaling/superglue_sample.csv", 
            "college_math": "data/scaling/gsm8k_sample.csv",  # Using GSM8K as college math proxy
            "humaneval": "humaneval"
        }
        
        # Models in order of parameter count for scaling analysis
        self.models = [
            "gpt-4o-mini",
            "claude-haiku", 
            "gpt-4o",
            "claude-sonnet",
            "llama-70b",
            "gpt-4",
            "claude-opus"
        ]
        
        self.total_cost_estimate = 0.0
        self.completed_experiments = []
        self.failed_experiments = []
    
    def estimate_total_cost(self) -> Dict[str, Any]:
        """Estimate total cost for the complete study."""
        cost_breakdown = {}
        total_cost = 0.0
        
        for model_key in self.models:
            model_costs = {}
            for dataset_name in self.datasets.keys():
                cost_info = estimate_experiment_cost(
                    model_key, 
                    self.sample_size,
                    avg_tokens_per_sample=2000,  # Rough estimate 
                    num_runs=self.num_runs
                )
                model_costs[dataset_name] = cost_info
                total_cost += cost_info["total_cost_usd"]
            
            cost_breakdown[model_key] = {
                "model_total": sum(c["total_cost_usd"] for c in model_costs.values()),
                "datasets": model_costs
            }
        
        self.total_cost_estimate = total_cost
        
        return {
            "total_estimated_cost": total_cost,
            "cost_breakdown": cost_breakdown,
            "experiment_matrix": {
                "models": len(self.models),
                "datasets": len(self.datasets), 
                "runs_per_combination": self.num_runs,
                "total_experiments": len(self.models) * len(self.datasets) * self.num_runs
            }
        }
    
    def create_experiment_config(
        self, 
        model_key: str, 
        dataset: str,
        run_id: int
    ) -> Dict[str, Any]:
        """Create configuration for a single experiment."""
        model_config = get_model_config(model_key)
        if not model_config:
            raise ValueError(f"Unknown model: {model_key}")
        
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%SZ")
        experiment_id = f"scaling_{model_key}_{dataset}_run{run_id}_{timestamp}"
        
        return {
            "experiment_id": experiment_id,
            "model": {
                "key": model_key,
                "name": model_config.name,
                "provider": model_config.provider,
                "api_model_name": model_config.api_model_name,
                "parameter_count_b": model_config.parameter_count_b,
                "size_category": model_config.size_category
            },
            "dataset": {
                "name": dataset,
                "path": self.datasets[dataset],
                "sample_size": self.sample_size
            },
            "settings": {
                "temperature": self.temperature,
                "max_turns": self.max_turns,
                "run_id": run_id,
                "total_runs": self.num_runs
            },
            "features": {
                "enable_multi_turn": True,  # Enable for all datasets per requirements
                "enable_error_awareness": True,
                "enable_confidence": True
            },
            "reproducibility": {
                "timestamp": timestamp,
                "git_commit": os.getenv("GIT_COMMIT", "unknown"),
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                "environment": {
                    "OPENAI_TEMPERATURE": str(self.temperature),
                    "MAX_TURNS": str(self.max_turns)
                }
            }
        }
    
    def run_single_experiment(
        self, 
        model_key: str,
        dataset: str,
        run_id: int,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """Run a single model+dataset+run combination."""
        
        config = self.create_experiment_config(model_key, dataset, run_id)
        experiment_id = config["experiment_id"]
        
        print(f"🔬 Running experiment: {experiment_id}")
        
        if dry_run:
            print(f"   [DRY RUN] Would run {model_key} on {dataset} (run {run_id})")
            return {"status": "dry_run", "config": config}
        
        # Set up environment
        env = os.environ.copy()
        env["OPENAI_TEMPERATURE"] = str(self.temperature)
        env["RUN_ID"] = experiment_id
        env["GIT_COMMIT"] = env.get("GIT_COMMIT", "scaling_study")
        
        # Determine subset parameter for HumanEval
        subset_param = []
        if dataset == "humaneval":
            subset_param = ["--subset", "subset_100"]  # Use 100 samples for HumanEval
        
        # Build command
        cmd = [
            sys.executable, "-m", "src.main", "run",
            "--dataset", self.datasets[dataset],
            "--provider", config["model"]["provider"],
            "--model", config["model"]["api_model_name"],
            "--max-turns", str(self.max_turns),
            "--out", str(self.output_dir / f"{experiment_id}_traces.json")
        ] + subset_param
        
        # Save experiment config
        config_file = self.output_dir / f"{experiment_id}_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        try:
            print(f"   Command: {' '.join(cmd)}")
            start_time = time.time()
            
            result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=3600)
            
            end_time = time.time()
            duration = end_time - start_time
            
            if result.returncode == 0:
                print(f"   ✅ Completed in {duration:.1f}s")
                
                return {
                    "status": "success",
                    "experiment_id": experiment_id,
                    "duration_s": duration,
                    "config": config,
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
            else:
                print(f"   ❌ Failed with code {result.returncode}")
                print(f"   Error: {result.stderr}")
                
                return {
                    "status": "failed",
                    "experiment_id": experiment_id,
                    "duration_s": duration,
                    "return_code": result.returncode,
                    "config": config,
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
                
        except subprocess.TimeoutExpired:
            print(f"   ⏱️ Experiment timed out after 1 hour")
            return {
                "status": "timeout",
                "experiment_id": experiment_id,
                "config": config
            }
        except Exception as e:
            print(f"   💥 Unexpected error: {e}")
            return {
                "status": "error",
                "experiment_id": experiment_id,
                "error": str(e),
                "config": config
            }
    
    def run_complete_study(
        self,
        models: List[str] = None,
        datasets: List[str] = None,
        dry_run: bool = False,
        max_parallel: int = 1
    ) -> Dict[str, Any]:
        """Run the complete scaling study."""
        
        if models is None:
            models = self.models
        if datasets is None:
            datasets = list(self.datasets.keys())
        
        print("🚀 Starting Scaling Laws Study")
        print("=" * 60)
        
        # Print cost estimate
        cost_info = self.estimate_total_cost()
        print(f"💰 Estimated total cost: ${cost_info['total_estimated_cost']:.2f}")
        print(f"🧪 Total experiments: {cost_info['experiment_matrix']['total_experiments']}")
        print()
        
        if not dry_run:
            response = input("Proceed with experiments? (y/N): ")
            if response.lower() != 'y':
                print("Aborted.")
                return {"status": "aborted"}
        
        study_start = time.time()
        
        # Run all experiments
        results = []
        total_experiments = len(models) * len(datasets) * self.num_runs
        completed = 0
        
        for model_key in models:
            for dataset in datasets:
                for run_id in range(1, self.num_runs + 1):
                    completed += 1
                    print(f"\n[{completed}/{total_experiments}] ", end="")
                    
                    result = self.run_single_experiment(model_key, dataset, run_id, dry_run)
                    results.append(result)
                    
                    if result["status"] == "success":
                        self.completed_experiments.append(result)
                    else:
                        self.failed_experiments.append(result)
                    
                    # Brief pause between experiments
                    if not dry_run:
                        time.sleep(2)
        
        study_duration = time.time() - study_start
        
        # Generate summary
        summary = {
            "study_metadata": {
                "start_time": datetime.fromtimestamp(study_start).isoformat(),
                "duration_s": study_duration,
                "models_tested": models,
                "datasets_tested": datasets,
                "runs_per_combination": self.num_runs
            },
            "results_summary": {
                "total_experiments": total_experiments,
                "successful": len(self.completed_experiments),
                "failed": len(self.failed_experiments),
                "success_rate": len(self.completed_experiments) / total_experiments
            },
            "cost_estimate": cost_info,
            "experiment_results": results,
            "output_directory": str(self.output_dir)
        }
        
        # Save complete results
        results_file = self.output_dir / "scaling_study_summary.json"
        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("\n" + "=" * 60)
        print(f"🎯 Study completed in {study_duration/3600:.1f} hours")
        print(f"✅ Successful experiments: {len(self.completed_experiments)}")
        print(f"❌ Failed experiments: {len(self.failed_experiments)}")
        print(f"📊 Results saved to: {results_file}")
        
        return summary


def main():
    parser = argparse.ArgumentParser(description="Run scaling laws study")
    parser.add_argument("--models", nargs="+", help="Models to test (default: all)")
    parser.add_argument("--datasets", nargs="+", help="Datasets to test (default: all)")
    parser.add_argument("--output-dir", default="scaling_study_results", help="Output directory")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be run without running")
    parser.add_argument("--sample-size", type=int, default=100, help="Sample size per dataset")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs per combination")
    parser.add_argument("--estimate-cost", action="store_true", help="Show cost estimate and exit")
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = ScalingStudyRunner(args.output_dir)
    runner.sample_size = args.sample_size
    runner.num_runs = args.runs
    
    if args.estimate_cost:
        cost_info = runner.estimate_total_cost()
        print("Scaling Study Cost Estimate")
        print("=" * 40)
        print(f"Total estimated cost: ${cost_info['total_estimated_cost']:.2f}")
        print(f"Total experiments: {cost_info['experiment_matrix']['total_experiments']}")
        print()
        
        print("Cost breakdown by model:")
        for model, costs in cost_info['cost_breakdown'].items():
            print(f"  {model}: ${costs['model_total']:.2f}")
        
        return
    
    # Run the study
    summary = runner.run_complete_study(
        models=args.models,
        datasets=args.datasets,
        dry_run=args.dry_run
    )
    
    if summary["results_summary"]["failed"] > 0:
        print("\nFailed experiments:")
        for failed in runner.failed_experiments:
            exp_id = failed.get('experiment_id', 'unknown')
            error_msg = failed.get('error', failed.get('status', 'Unknown error'))
            print(f"  - {exp_id}: {error_msg}")
    
    print(f"\nNext steps:")
    print(f"1. Run analysis: python -m src.scaling.analysis {args.output_dir}")
    print(f"2. Generate visualizations: python scripts/generate_scaling_plots.py {args.output_dir}")
    print(f"3. Create publication tables: python scripts/generate_publication_artifacts.py {args.output_dir}")


if __name__ == "__main__":
    main()