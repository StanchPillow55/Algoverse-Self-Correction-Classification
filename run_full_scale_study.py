#!/usr/bin/env python3
"""
Full-Scale Scaling Study - Real Datasets
Maximum 1000 questions per dataset, all models, comprehensive analysis
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
import tempfile
import csv

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.data.scaling_datasets import ScalingDatasetManager
from src.scaling.model_registry import MODEL_REGISTRY, get_model_config, estimate_experiment_cost
from src.loop.runner import run_dataset

class FullScaleStudyRunner:
    """Runs comprehensive scaling study on real datasets."""
    
    def __init__(self, output_dir: str = "full_scale_study_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Study parameters
        self.max_questions_per_dataset = 1000
        self.max_turns = 3
        self.temperature = 0.2
        
        # Real datasets configuration
        self.datasets = {
            "gsm8k": {"type": "math", "max_samples": 1000},
            "humaneval": {"type": "code", "max_samples": 164},  # Full dataset is only 164
            "superglue": {"type": "reasoning", "max_samples": 1000},
            "mathbench": {"type": "math_advanced", "max_samples": 1000}
        }
        
        # Models in scaling order
        self.models = [
            "gpt-4o-mini",    # 1.8B
            "claude-haiku",   # 3.0B  
            "gpt-4o",         # 8.0B
            "claude-sonnet",  # 70B
            "llama-70b",      # 70B
            "gpt-4",          # 175B
            "claude-opus"     # 175B
        ]
        
        # Initialize dataset manager
        self.dm = ScalingDatasetManager()
        
        # Results storage
        self.results = {
            "experiment_metadata": {
                "start_time": datetime.now().isoformat(),
                "datasets": self.datasets,
                "models": self.models,
                "max_questions_per_dataset": self.max_questions_per_dataset,
                "max_turns": self.max_turns,
                "temperature": self.temperature
            },
            "experiments": [],
            "cost_tracking": [],
            "summary_statistics": {}
        }
        
    def download_all_datasets(self) -> bool:
        """Download all required datasets."""
        print("📥 DOWNLOADING ALL DATASETS...")
        print("=" * 50)
        
        success = True
        for dataset_name in self.datasets.keys():
            print(f"\nDownloading {dataset_name}...")
            
            if not self.dm.download_dataset(dataset_name, force=False):
                print(f"❌ Failed to download {dataset_name}")
                success = False
            else:
                info = self.dm.get_dataset_info(dataset_name)
                print(f"✅ {dataset_name}: {info['total_samples']} samples available")
        
        return success
    
    def estimate_total_cost(self) -> Dict[str, Any]:
        """Estimate costs for the full study."""
        print("\n💰 COST ESTIMATION...")
        print("=" * 50)
        
        total_cost = 0.0
        cost_breakdown = {}
        
        for model_key in self.models:
            model_cost = 0.0
            model_breakdown = {}
            
            for dataset_name, config in self.datasets.items():
                max_samples = min(config["max_samples"], self.max_questions_per_dataset)
                
                try:
                    cost_info = estimate_experiment_cost(
                        model_key,
                        max_samples,
                        avg_tokens_per_sample=1500,  # Conservative estimate
                        num_runs=1
                    )
                    
                    model_breakdown[dataset_name] = cost_info
                    model_cost += cost_info["total_cost_usd"]
                    
                except Exception as e:
                    print(f"⚠️ Could not estimate cost for {model_key}: {e}")
                    model_breakdown[dataset_name] = {"total_cost_usd": 0.0}
            
            cost_breakdown[model_key] = {
                "model_total_usd": model_cost,
                "datasets": model_breakdown
            }
            total_cost += model_cost
            
            print(f"  {model_key:<15} ${model_cost:>8.2f}")
        
        print(f"\n💳 TOTAL ESTIMATED COST: ${total_cost:.2f}")
        
        cost_summary = {
            "total_estimated_cost_usd": total_cost,
            "cost_breakdown": cost_breakdown,
            "total_experiments": len(self.models) * len(self.datasets),
            "total_samples": sum(min(c["max_samples"], self.max_questions_per_dataset) 
                               for c in self.datasets.values()) * len(self.models)
        }
        
        # Save cost estimate
        with open(self.output_dir / "cost_estimate.json", 'w') as f:
            json.dump(cost_summary, f, indent=2)
        
        return cost_summary
    
    def prepare_dataset_csv(self, dataset_name: str, max_samples: int) -> str:
        """Prepare a CSV file for a dataset with the specified sample limit."""
        
        # Load samples from the dataset
        samples = self.dm.load_dataset(dataset_name, sample_size=max_samples)
        
        if not samples:
            raise ValueError(f"No samples loaded for {dataset_name}")
        
        # Create temporary CSV file
        temp_file = tempfile.NamedTemporaryFile(
            mode='w', 
            suffix=f'_{dataset_name}.csv', 
            delete=False,
            dir=str(self.output_dir)
        )
        
        # Write CSV with standard format
        fieldnames = ['qid', 'question', 'ground_truth', 'topic']
        writer = csv.DictWriter(temp_file, fieldnames=fieldnames)
        writer.writeheader()
        
        for i, sample in enumerate(samples):
            writer.writerow({
                'qid': sample.get('id', f'{dataset_name}_{i}'),
                'question': sample['question'],
                'ground_truth': sample['answer'],
                'topic': dataset_name
            })
        
        temp_file.close()
        print(f"  📄 Created {dataset_name} CSV: {len(samples)} samples -> {temp_file.name}")
        
        return temp_file.name
    
    def run_single_experiment(
        self, 
        model_key: str, 
        dataset_name: str, 
        dataset_config: Dict[str, Any],
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """Run a single model+dataset combination."""
        
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%SZ")
        experiment_id = f"fullscale_{model_key}_{dataset_name}_{timestamp}"
        
        print(f"\n🔬 EXPERIMENT: {experiment_id}")
        print("-" * 60)
        
        # Get model configuration
        model_config = get_model_config(model_key)
        if not model_config:
            raise ValueError(f"Unknown model: {model_key}")
        
        print(f"  Model: {model_config.name} ({model_config.parameter_count_b}B params)")
        print(f"  Dataset: {dataset_name} ({dataset_config['type']})")
        
        max_samples = min(dataset_config["max_samples"], self.max_questions_per_dataset)
        print(f"  Max samples: {max_samples}")
        
        if dry_run:
            return {
                "experiment_id": experiment_id,
                "model": model_key,
                "dataset": dataset_name,
                "status": "dry_run",
                "max_samples": max_samples
            }
        
        try:
            # Prepare dataset CSV
            dataset_csv = self.prepare_dataset_csv(dataset_name, max_samples)
            
            # Set up environment
            os.environ["OPENAI_TEMPERATURE"] = str(self.temperature)
            os.environ["RUN_ID"] = experiment_id
            
            # Create traces output path
            traces_output = self.output_dir / f"{experiment_id}_traces.json"
            
            print(f"  🚀 Running {model_config.provider} {model_config.api_model_name}...")
            
            start_time = time.time()
            
            # Run the experiment using the runner
            results = run_dataset(
                dataset_csv=dataset_csv,
                traces_out=str(traces_output),
                max_turns=self.max_turns,
                provider=model_config.provider,
                model=model_config.api_model_name,
                experiment_id=experiment_id,
                dataset_name=dataset_name
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Clean up temporary CSV
            try:
                os.unlink(dataset_csv)
            except:
                pass
            
            # Process results
            experiment_result = {
                "experiment_id": experiment_id,
                "model": {
                    "key": model_key,
                    "name": model_config.name,
                    "provider": model_config.provider,
                    "parameter_count_b": model_config.parameter_count_b,
                    "size_category": model_config.size_category
                },
                "dataset": {
                    "name": dataset_name,
                    "type": dataset_config["type"],
                    "max_samples": max_samples
                },
                "settings": {
                    "temperature": self.temperature,
                    "max_turns": self.max_turns
                },
                "results": results,
                "execution": {
                    "duration_seconds": duration,
                    "timestamp": timestamp,
                    "traces_file": str(traces_output)
                },
                "status": "completed"
            }
            
            print(f"  ✅ Completed in {duration:.1f}s")
            if results and 'summary' in results:
                summary = results['summary']
                print(f"     📊 Accuracy: {summary.get('final_accuracy_mean', 0):.3f}")
                print(f"     📄 Samples: {summary.get('items', 0)}")
            
            return experiment_result
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            
            return {
                "experiment_id": experiment_id,
                "model": {"key": model_key, "name": model_config.name},
                "dataset": {"name": dataset_name},
                "status": "failed",
                "error": str(e),
                "timestamp": timestamp
            }
    
    def run_full_study(self, dry_run: bool = False, specific_models: List[str] = None, specific_datasets: List[str] = None) -> Dict[str, Any]:
        """Run the complete scaling study."""
        
        print("🚀 FULL-SCALE SCALING STUDY")
        print("=" * 60)
        print(f"Max questions per dataset: {self.max_questions_per_dataset}")
        print(f"Max turns per question: {self.max_turns}")
        print(f"Models: {len(self.models)}")
        print(f"Datasets: {len(self.datasets)}")
        print(f"Total experiments: {len(self.models) * len(self.datasets)}")
        
        # Filter models and datasets if specified
        models_to_run = specific_models if specific_models else self.models
        datasets_to_run = {k: v for k, v in self.datasets.items() 
                          if not specific_datasets or k in specific_datasets}
        
        print(f"\nRunning: {len(models_to_run)} models × {len(datasets_to_run)} datasets")
        
        if not dry_run:
            # Download datasets
            if not self.download_all_datasets():
                print("❌ Dataset download failed. Aborting.")
                return {"status": "failed", "reason": "dataset_download_failed"}
        
        # Cost estimation
        cost_info = self.estimate_total_cost()
        
        if not dry_run:
            print(f"\n⚠️  This will cost approximately ${cost_info['total_estimated_cost_usd']:.2f}")
            response = input("Continue? (y/N): ")
            if response.lower() != 'y':
                print("Study cancelled.")
                return {"status": "cancelled"}
        
        # Run experiments
        print(f"\n🔬 RUNNING EXPERIMENTS...")
        print("=" * 60)
        
        study_start_time = time.time()
        completed = 0
        failed = 0
        
        for i, model_key in enumerate(models_to_run):
            for j, (dataset_name, dataset_config) in enumerate(datasets_to_run.items()):
                
                experiment_num = i * len(datasets_to_run) + j + 1
                total_experiments = len(models_to_run) * len(datasets_to_run)
                
                print(f"\n[{experiment_num}/{total_experiments}] {model_key} × {dataset_name}")
                
                result = self.run_single_experiment(
                    model_key, dataset_name, dataset_config, dry_run
                )
                
                self.results["experiments"].append(result)
                
                if result["status"] == "completed":
                    completed += 1
                else:
                    failed += 1
                
                # Save intermediate results
                if not dry_run and experiment_num % 5 == 0:
                    self.save_results()
        
        study_end_time = time.time()
        study_duration = study_end_time - study_start_time
        
        # Final results
        self.results["experiment_metadata"]["end_time"] = datetime.now().isoformat()
        self.results["experiment_metadata"]["total_duration_seconds"] = study_duration
        
        self.results["summary_statistics"] = {
            "total_experiments": len(self.results["experiments"]),
            "completed": completed,
            "failed": failed,
            "success_rate": completed / len(self.results["experiments"]) if self.results["experiments"] else 0,
            "total_duration_hours": study_duration / 3600,
            "avg_experiment_duration_minutes": (study_duration / len(self.results["experiments"]) / 60) if self.results["experiments"] else 0
        }
        
        # Save final results
        self.save_results()
        
        # Print final summary
        self.print_final_summary()
        
        return self.results
    
    def save_results(self):
        """Save current results to files."""
        
        # Main results file
        results_file = self.output_dir / "full_scale_study_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"  💾 Results saved: {results_file}")
    
    def print_final_summary(self):
        """Print comprehensive final summary."""
        
        print(f"\n🎉 FULL-SCALE STUDY COMPLETE!")
        print("=" * 60)
        
        stats = self.results["summary_statistics"]
        print(f"📊 SUMMARY STATISTICS:")
        print(f"  Total experiments: {stats['total_experiments']}")
        print(f"  Completed: {stats['completed']}")
        print(f"  Failed: {stats['failed']}")
        print(f"  Success rate: {stats['success_rate']:.1%}")
        print(f"  Total duration: {stats['total_duration_hours']:.1f} hours")
        print(f"  Avg per experiment: {stats['avg_experiment_duration_minutes']:.1f} minutes")
        
        # Results by model
        print(f"\n📈 RESULTS BY MODEL:")
        model_results = {}
        for exp in self.results["experiments"]:
            if exp["status"] == "completed":
                model_key = exp["model"]["key"]
                if model_key not in model_results:
                    model_results[model_key] = {
                        "experiments": 0,
                        "total_accuracy": 0,
                        "parameter_count": exp["model"]["parameter_count_b"]
                    }
                model_results[model_key]["experiments"] += 1
                if "results" in exp and "summary" in exp["results"]:
                    model_results[model_key]["total_accuracy"] += exp["results"]["summary"].get("final_accuracy_mean", 0)
        
        for model_key, data in sorted(model_results.items(), key=lambda x: x[1]["parameter_count"]):
            avg_accuracy = data["total_accuracy"] / data["experiments"] if data["experiments"] > 0 else 0
            print(f"  {model_key:<15} {data['parameter_count']:>6.1f}B  {avg_accuracy:>6.3f} avg accuracy")
        
        print(f"\n📁 Results saved in: {self.output_dir}")
        print(f"🔬 Ready for scaling law analysis!")

def main():
    parser = argparse.ArgumentParser(description="Run full-scale scaling study")
    parser.add_argument("--dry-run", action="store_true", help="Dry run (no actual experiments)")
    parser.add_argument("--models", nargs="*", help="Specific models to run")
    parser.add_argument("--datasets", nargs="*", help="Specific datasets to run")
    parser.add_argument("--output", default="full_scale_study_results", help="Output directory")
    
    args = parser.parse_args()
    
    runner = FullScaleStudyRunner(output_dir=args.output)
    
    results = runner.run_full_study(
        dry_run=args.dry_run,
        specific_models=args.models,
        specific_datasets=args.datasets
    )
    
    if results.get("status") == "cancelled":
        sys.exit(1)
    
    success = results.get("summary_statistics", {}).get("success_rate", 0) > 0.5
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()