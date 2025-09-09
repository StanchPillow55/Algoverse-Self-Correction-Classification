#!/usr/bin/env python3
"""
David's Analysis Runner for Scaling Study

Runs comprehensive analysis on Phase 1 results and prepares for Phase 2/3.
"""

import sys
import json
import time
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.trace_formatter import TraceFormatter
from src.utils.scaling_analyzer import ScalingAnalyzer
from src.utils.result_aggregator import ResultAggregator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DavidAnalysisRunner:
    """Runs David's analysis tasks for the scaling study."""
    
    def __init__(self, output_dir: str = "outputs/david_analysis"):
        """Initialize analysis runner."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.trace_formatter = TraceFormatter(str(self.output_dir / "formatted_traces"))
        self.scaling_analyzer = ScalingAnalyzer()
        self.result_aggregator = ResultAggregator()
    
    def analyze_phase1_results(self) -> Dict[str, Any]:
        """Analyze Phase 1 validation results."""
        logger.info("📊 Analyzing Phase 1 Results")
        
        # Load Phase 1 results
        phase1_file = Path("outputs/phase1_simple/phase1_analysis.json")
        if not phase1_file.exists():
            logger.error("Phase 1 results not found. Run Phase 1 first.")
            return {}
        
        with open(phase1_file, 'r') as f:
            phase1_data = json.load(f)
        
        results = phase1_data.get('results', [])
        if not results:
            logger.error("No Phase 1 results found.")
            return {}
        
        # Create analysis results
        analysis = {
            "phase": "phase1_analysis",
            "timestamp": time.time(),
            "models_analyzed": len(results),
            "model_performance": {},
            "scaling_insights": {},
            "cost_analysis": {},
            "statistical_analysis": {}
        }
        
        # Analyze each model
        for result in results:
            model = result['model']
            accuracy = result['accuracy']
            total_samples = result['total_samples']
            correct_samples = result['correct_samples']
            
            analysis["model_performance"][model] = {
                "accuracy": accuracy,
                "total_samples": total_samples,
                "correct_samples": correct_samples,
                "error_rate": 1 - accuracy
            }
            
            logger.info(f"   {model}: {accuracy:.3f} accuracy ({correct_samples}/{total_samples})")
        
        # Calculate scaling insights
        if len(results) >= 2:
            models = [r['model'] for r in results]
            accuracies = [r['accuracy'] for r in results]
            
            # Simple scaling analysis (model size vs accuracy)
            analysis["scaling_insights"] = {
                "model_count": len(models),
                "accuracy_range": {
                    "min": min(accuracies),
                    "max": max(accuracies),
                    "mean": np.mean(accuracies),
                    "std": np.std(accuracies)
                },
                "performance_gap": max(accuracies) - min(accuracies)
            }
        
        # Cost analysis (placeholder - would need actual cost data)
        analysis["cost_analysis"] = {
            "note": "Cost analysis requires actual API usage data",
            "models": {r['model']: "Cost data not available" for r in results}
        }
        
        # Statistical analysis
        if len(results) >= 2:
            analysis["statistical_analysis"] = {
                "sample_sizes": [r['total_samples'] for r in results],
                "accuracy_means": [r['accuracy'] for r in results],
                "confidence_intervals": self._calculate_confidence_intervals(results),
                "significance_tests": self._run_significance_tests(results)
            }
        
        # Save analysis
        analysis_file = self.output_dir / "phase1_analysis.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        logger.info(f"📊 Phase 1 analysis saved to: {analysis_file}")
        
        return analysis
    
    def _calculate_confidence_intervals(self, results: List[Dict]) -> Dict[str, Any]:
        """Calculate 95% confidence intervals for accuracy metrics."""
        ci_results = {}
        
        for result in results:
            model = result['model']
            accuracy = result['accuracy']
            n = result['total_samples']
            
            # Calculate 95% CI using normal approximation
            se = np.sqrt(accuracy * (1 - accuracy) / n)
            ci_lower = max(0, accuracy - 1.96 * se)
            ci_upper = min(1, accuracy + 1.96 * se)
            
            ci_results[model] = {
                "accuracy": accuracy,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "margin_of_error": 1.96 * se
            }
        
        return ci_results
    
    def _run_significance_tests(self, results: List[Dict]) -> Dict[str, Any]:
        """Run significance tests between models."""
        if len(results) < 2:
            return {"note": "Need at least 2 models for significance testing"}
        
        # Simple t-test simulation (would need scipy for real implementation)
        models = [r['model'] for r in results]
        accuracies = [r['accuracy'] for r in results]
        
        # Calculate basic statistics
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        
        return {
            "models": models,
            "mean_accuracy": mean_acc,
            "std_accuracy": std_acc,
            "note": "Full significance testing requires scipy.stats implementation"
        }
    
    def prepare_phase2_experiments(self) -> Dict[str, Any]:
        """Prepare Phase 2 medium scale experiments."""
        logger.info("🚀 Preparing Phase 2 Medium Scale Experiments")
        
        phase2_config = {
            "phase": "phase2_medium_scale",
            "description": "4 models × 2 datasets × 500 samples",
            "models": ["gpt-4o-mini", "claude-haiku", "gpt-4o", "claude-sonnet"],
            "datasets": ["gsm8k", "humaneval"],
            "sample_sizes": [100, 500],
            "max_turns": 3,
            "estimated_duration": "2-3 hours",
            "estimated_cost": "~$50-100"
        }
        
        # Create experiment scripts
        self._create_phase2_scripts(phase2_config)
        
        # Save configuration
        config_file = self.output_dir / "phase2_config.json"
        with open(config_file, 'w') as f:
            json.dump(phase2_config, f, indent=2)
        
        logger.info(f"📋 Phase 2 configuration saved to: {config_file}")
        
        return phase2_config
    
    def _create_phase2_scripts(self, config: Dict[str, Any]):
        """Create scripts for Phase 2 experiments."""
        script_dir = self.output_dir / "phase2_scripts"
        script_dir.mkdir(exist_ok=True)
        
        # Create individual experiment scripts
        for model in config["models"]:
            for dataset in config["datasets"]:
                for sample_size in config["sample_sizes"]:
                    script_name = f"run_{model}_{dataset}_{sample_size}.sh"
                    script_path = script_dir / script_name
                    
                    script_content = f"""#!/bin/bash
# Phase 2 Experiment: {model} on {dataset} ({sample_size} samples)

echo "🚀 Running Phase 2: {model} on {dataset} ({sample_size} samples)"

# Set environment
export PYTHONPATH=/Users/bradleyharaguchi/Algoverse-Self-Correction-Classification

# Determine provider
if [[ "{model}" == *"gpt"* ]]; then
    PROVIDER="openai"
elif [[ "{model}" == *"claude"* ]]; then
    PROVIDER="anthropic"
else
    PROVIDER="openai"
fi

# Run experiment
python src/main.py run \\
    --dataset data/{dataset}/test_{sample_size}.jsonl \\
    --out outputs/phase2_medium/{model}_{dataset}_{sample_size}/traces.jsonl \\
    --max-turns 3 \\
    --provider $PROVIDER \\
    --model {model}

echo "✅ Experiment completed: {model} on {dataset} ({sample_size} samples)"
"""
                    
                    with open(script_path, 'w') as f:
                        f.write(script_content)
                    
                    # Make executable
                    script_path.chmod(0o755)
        
        # Create master script
        master_script = script_dir / "run_all_phase2.sh"
        master_content = """#!/bin/bash
# Master script to run all Phase 2 experiments

echo "🚀 Starting Phase 2 Medium Scale Experiments"
echo "=============================================="

# Run all individual experiments
for script in *.sh; do
    if [[ "$script" != "run_all_phase2.sh" ]]; then
        echo "Running $script..."
        ./$script
        echo "Completed $script"
        echo "---"
    fi
done

echo "✅ All Phase 2 experiments completed!"
"""
        
        with open(master_script, 'w') as f:
            f.write(master_content)
        
        master_script.chmod(0o755)
        
        logger.info(f"📝 Phase 2 scripts created in: {script_dir}")
    
    def generate_scaling_visualizations(self) -> Dict[str, Any]:
        """Generate scaling law visualizations."""
        logger.info("📈 Generating Scaling Law Visualizations")
        
        # Load Phase 1 results
        phase1_file = Path("outputs/phase1_simple/phase1_analysis.json")
        if not phase1_file.exists():
            logger.error("Phase 1 results not found. Run Phase 1 first.")
            return {}
        
        with open(phase1_file, 'r') as f:
            phase1_data = json.load(f)
        
        results = phase1_data.get('results', [])
        if not results:
            logger.error("No Phase 1 results found.")
            return {}
        
        # Create visualization data
        viz_data = {
            "models": [r['model'] for r in results],
            "accuracies": [r['accuracy'] for r in results],
            "sample_sizes": [r['total_samples'] for r in results],
            "correct_samples": [r['correct_samples'] for r in results]
        }
        
        # Create simple visualization (would use matplotlib in real implementation)
        viz_file = self.output_dir / "scaling_visualization.json"
        with open(viz_file, 'w') as f:
            json.dump(viz_data, f, indent=2)
        
        logger.info(f"📈 Visualization data saved to: {viz_file}")
        
        return viz_data
    
    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """Run all analysis tasks."""
        logger.info("🎯 Running Comprehensive Analysis for David's Tasks")
        logger.info("=" * 60)
        
        all_results = {
            "start_time": time.time(),
            "phase1_analysis": {},
            "phase2_preparation": {},
            "visualizations": {},
            "summary": {}
        }
        
        # Run Phase 1 analysis
        try:
            all_results["phase1_analysis"] = self.analyze_phase1_results()
            logger.info("✅ Phase 1 analysis completed")
        except Exception as e:
            logger.error(f"❌ Phase 1 analysis failed: {e}")
            all_results["phase1_analysis"] = {"error": str(e)}
        
        # Prepare Phase 2
        try:
            all_results["phase2_preparation"] = self.prepare_phase2_experiments()
            logger.info("✅ Phase 2 preparation completed")
        except Exception as e:
            logger.error(f"❌ Phase 2 preparation failed: {e}")
            all_results["phase2_preparation"] = {"error": str(e)}
        
        # Generate visualizations
        try:
            all_results["visualizations"] = self.generate_scaling_visualizations()
            logger.info("✅ Visualizations generated")
        except Exception as e:
            logger.error(f"❌ Visualization generation failed: {e}")
            all_results["visualizations"] = {"error": str(e)}
        
        # Generate summary
        all_results["end_time"] = time.time()
        all_results["duration"] = all_results["end_time"] - all_results["start_time"]
        
        all_results["summary"] = {
            "total_duration": all_results["duration"],
            "tasks_completed": len([k for k, v in all_results.items() if isinstance(v, dict) and "error" not in v]),
            "tasks_failed": len([k for k, v in all_results.items() if isinstance(v, dict) and "error" in v]),
            "next_steps": [
                "Run Phase 2 experiments using generated scripts",
                "Collect cost data from API usage",
                "Implement full statistical analysis with scipy",
                "Generate matplotlib visualizations",
                "Prepare Phase 3 full scale experiments"
            ]
        }
        
        # Save all results
        results_file = self.output_dir / "comprehensive_analysis.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        logger.info(f"🎉 Comprehensive analysis completed in {all_results['duration']:.1f}s")
        logger.info(f"📊 Results saved to: {results_file}")
        
        return all_results

def main():
    """Main function to run David's analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run David's scaling study analysis")
    parser.add_argument("--task", choices=["phase1", "phase2", "visualizations", "all"], 
                       default="all", help="Which analysis task to run")
    parser.add_argument("--output-dir", default="outputs/david_analysis",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = DavidAnalysisRunner(args.output_dir)
    
    if args.task == "phase1":
        runner.analyze_phase1_results()
    elif args.task == "phase2":
        runner.prepare_phase2_experiments()
    elif args.task == "visualizations":
        runner.generate_scaling_visualizations()
    elif args.task == "all":
        runner.run_comprehensive_analysis()

if __name__ == "__main__":
    main()
