#!/usr/bin/env python3
"""
David's Experiment Runner for Scaling Study

Runs all phases of the scaling study experiments with proper trace formatting.
"""

import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.experiments.scaling_runner import ScalingExperimentRunner, ScalingExperimentConfig
from src.utils.trace_formatter import TraceFormatter
from src.utils.scaling_analyzer import ScalingAnalyzer
from src.utils.result_aggregator import ResultAggregator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DavidExperimentRunner:
    """Runs David's assigned experiments for the scaling study."""
    
    def __init__(self, output_dir: str = "outputs/david_experiments"):
        """Initialize experiment runner."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.trace_formatter = TraceFormatter(str(self.output_dir / "formatted_traces"))
        self.scaling_analyzer = ScalingAnalyzer()
        self.result_aggregator = ResultAggregator()
        
        # Experiment phases
        self.phases = {
            "phase_1": {
                "name": "Phase 1: Validation",
                "models": ["gpt-4o-mini", "claude-haiku"],
                "datasets": ["gsm8k"],
                "sample_sizes": [100],
                "max_turns": 3,
                "description": "2 models × 1 dataset × 100 samples - Validate pipeline"
            },
            "phase_2": {
                "name": "Phase 2: Medium Scale",
                "models": ["gpt-4o-mini", "claude-haiku", "gpt-4o", "claude-sonnet"],
                "datasets": ["gsm8k", "humaneval"],
                "sample_sizes": [100, 500],
                "max_turns": 3,
                "description": "4 models × 2 datasets × 500 samples - Test scaling hypothesis"
            },
            "phase_3": {
                "name": "Phase 3: Full Scale",
                "models": ["gpt-4o-mini", "claude-haiku", "gpt-4o", "claude-sonnet", "llama-70b", "gpt-4", "claude-opus"],
                "datasets": ["gsm8k", "humaneval", "toolqa", "mathbench"],
                "sample_sizes": [100, 500, 1000],
                "max_turns": 3,
                "description": "7 models × 4 datasets × 1000 samples - Complete scaling analysis"
            }
        }
    
    def run_phase(self, phase_name: str, dry_run: bool = False) -> Dict[str, Any]:
        """Run a specific phase of experiments."""
        if phase_name not in self.phases:
            raise ValueError(f"Unknown phase: {phase_name}")
        
        phase_config = self.phases[phase_name]
        logger.info(f"🚀 Starting {phase_config['name']}")
        logger.info(f"   {phase_config['description']}")
        
        phase_results = {
            "phase": phase_name,
            "config": phase_config,
            "start_time": time.time(),
            "experiments": [],
            "formatted_traces": {},
            "analysis": {}
        }
        
        # Create phase output directory
        phase_dir = self.output_dir / phase_name
        phase_dir.mkdir(exist_ok=True)
        
        # Run experiments for each model-dataset-sample_size combination
        total_experiments = 0
        for model in phase_config["models"]:
            for dataset in phase_config["datasets"]:
                for sample_size in phase_config["sample_sizes"]:
                    total_experiments += 1
        
        logger.info(f"   Total experiments to run: {total_experiments}")
        
        experiment_count = 0
        for model in phase_config["models"]:
            for dataset in phase_config["datasets"]:
                for sample_size in phase_config["sample_sizes"]:
                    experiment_count += 1
                    
                    experiment_id = f"{phase_name}_{model}_{dataset}_{sample_size}"
                    logger.info(f"   [{experiment_count}/{total_experiments}] Running: {experiment_id}")
                    
                    if dry_run:
                        logger.info(f"   [DRY RUN] Would run {experiment_id}")
                        continue
                    
                    # Run experiment
                    experiment_result = self._run_single_experiment(
                        experiment_id, model, dataset, sample_size, 
                        phase_config["max_turns"], phase_dir
                    )
                    
                    if experiment_result:
                        phase_results["experiments"].append(experiment_result)
                        
                        # Format traces
                        if "traces_file" in experiment_result:
                            formatted = self.trace_formatter.format_experiment_traces(
                                experiment_result["traces_file"], experiment_id
                            )
                            phase_results["formatted_traces"][experiment_id] = formatted
        
        phase_results["end_time"] = time.time()
        phase_results["duration"] = phase_results["end_time"] - phase_results["start_time"]
        
        # Save phase results
        phase_file = phase_dir / "phase_results.json"
        with open(phase_file, 'w') as f:
            json.dump(phase_results, f, indent=2)
        
        logger.info(f"✅ {phase_config['name']} completed in {phase_results['duration']:.1f}s")
        logger.info(f"   Results saved to: {phase_file}")
        
        return phase_results
    
    def _run_single_experiment(self, experiment_id: str, model: str, dataset: str, 
                              sample_size: int, max_turns: int, output_dir: Path) -> Optional[Dict[str, Any]]:
        """Run a single experiment."""
        try:
            # Create experiment configuration
            config = ScalingExperimentConfig(
                models=[model],
                datasets=[dataset],
                sample_sizes=[sample_size],
                max_turns=max_turns,
                temperature=0.2,
                output_dir=str(output_dir / experiment_id)
            )
            
            # Run experiment
            runner = ScalingExperimentRunner(config)
            results = runner.run_experiments()
            
            if results and len(results) > 0:
                experiment_result = results[0]  # Get first (and only) result
                experiment_result["experiment_id"] = experiment_id
                experiment_result["model"] = model
                experiment_result["dataset"] = dataset
                experiment_result["sample_size"] = sample_size
                
                return experiment_result
            else:
                logger.error(f"   ❌ No results from {experiment_id}")
                return None
                
        except Exception as e:
            logger.error(f"   ❌ Error running {experiment_id}: {e}")
            return None
    
    def run_all_phases(self, dry_run: bool = False) -> Dict[str, Any]:
        """Run all phases of experiments."""
        logger.info("🎯 Starting David's Scaling Study Experiments")
        logger.info("=" * 60)
        
        all_results = {
            "start_time": time.time(),
            "phases": {},
            "summary": {}
        }
        
        # Run each phase
        for phase_name in ["phase_1", "phase_2", "phase_3"]:
            try:
                phase_results = self.run_phase(phase_name, dry_run)
                all_results["phases"][phase_name] = phase_results
                
                # Brief pause between phases
                if not dry_run:
                    time.sleep(2)
                    
            except Exception as e:
                logger.error(f"❌ Phase {phase_name} failed: {e}")
                all_results["phases"][phase_name] = {"error": str(e)}
        
        all_results["end_time"] = time.time()
        all_results["total_duration"] = all_results["end_time"] - all_results["start_time"]
        
        # Generate summary
        all_results["summary"] = self._generate_summary(all_results)
        
        # Save all results
        results_file = self.output_dir / "all_experiments_results.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        logger.info(f"🎉 All experiments completed in {all_results['total_duration']:.1f}s")
        logger.info(f"   Results saved to: {results_file}")
        
        return all_results
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of all experiments."""
        summary = {
            "total_phases": len(results["phases"]),
            "successful_phases": 0,
            "total_experiments": 0,
            "successful_experiments": 0,
            "total_duration": results["total_duration"],
            "phases_summary": {}
        }
        
        for phase_name, phase_data in results["phases"].items():
            if "error" in phase_data:
                summary["phases_summary"][phase_name] = {"status": "failed", "error": phase_data["error"]}
                continue
            
            summary["successful_phases"] += 1
            experiments = phase_data.get("experiments", [])
            summary["total_experiments"] += len(experiments)
            summary["successful_experiments"] += len(experiments)
            
            summary["phases_summary"][phase_name] = {
                "status": "completed",
                "experiments": len(experiments),
                "duration": phase_data.get("duration", 0),
                "models": phase_data["config"]["models"],
                "datasets": phase_data["config"]["datasets"],
                "sample_sizes": phase_data["config"]["sample_sizes"]
            }
        
        return summary
    
    def analyze_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze results from all experiments."""
        logger.info("📊 Analyzing experiment results")
        
        analysis_results = {
            "scaling_analysis": {},
            "cost_benefit_analysis": {},
            "statistical_analysis": {},
            "visualizations": {}
        }
        
        # Collect all experiment results for analysis
        all_experiments = []
        for phase_name, phase_data in results["phases"].items():
            if "experiments" in phase_data:
                all_experiments.extend(phase_data["experiments"])
        
        if not all_experiments:
            logger.warning("No experiments found for analysis")
            return analysis_results
        
        # Run scaling analysis
        try:
            scaling_results = self.scaling_analyzer.analyze_scaling(all_experiments)
            analysis_results["scaling_analysis"] = scaling_results
            logger.info("✅ Scaling analysis completed")
        except Exception as e:
            logger.error(f"❌ Scaling analysis failed: {e}")
            analysis_results["scaling_analysis"] = {"error": str(e)}
        
        # Run cost-benefit analysis
        try:
            cost_benefit = self.result_aggregator.calculate_cost_benefit_ratios()
            analysis_results["cost_benefit_analysis"] = cost_benefit
            logger.info("✅ Cost-benefit analysis completed")
        except Exception as e:
            logger.error(f"❌ Cost-benefit analysis failed: {e}")
            analysis_results["cost_benefit_analysis"] = {"error": str(e)}
        
        # Run statistical analysis
        try:
            stats = self.result_aggregator.calculate_statistical_significance()
            analysis_results["statistical_analysis"] = stats
            logger.info("✅ Statistical analysis completed")
        except Exception as e:
            logger.error(f"❌ Statistical analysis failed: {e}")
            analysis_results["statistical_analysis"] = {"error": str(e)}
        
        # Save analysis results
        analysis_file = self.output_dir / "analysis_results.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis_results, f, indent=2)
        
        logger.info(f"📊 Analysis results saved to: {analysis_file}")
        
        return analysis_results

def main():
    """Main function to run David's experiments."""
    parser = argparse.ArgumentParser(description="Run David's scaling study experiments")
    parser.add_argument("--phase", choices=["phase_1", "phase_2", "phase_3", "all"], 
                       default="all", help="Which phase to run")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Show what would be run without actually running")
    parser.add_argument("--output-dir", default="outputs/david_experiments",
                       help="Output directory for results")
    parser.add_argument("--analyze", action="store_true",
                       help="Run analysis on existing results")
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = DavidExperimentRunner(args.output_dir)
    
    if args.analyze:
        # Load existing results and analyze
        results_file = Path(args.output_dir) / "all_experiments_results.json"
        if results_file.exists():
            with open(results_file, 'r') as f:
                results = json.load(f)
            runner.analyze_results(results)
        else:
            logger.error(f"No results file found: {results_file}")
    else:
        # Run experiments
        if args.phase == "all":
            results = runner.run_all_phases(args.dry_run)
            if not args.dry_run:
                runner.analyze_results(results)
        else:
            results = runner.run_phase(args.phase, args.dry_run)

if __name__ == "__main__":
    main()
