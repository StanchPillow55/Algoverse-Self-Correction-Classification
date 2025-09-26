#!/usr/bin/env python3
"""
Enhanced Anthropic Multi-Turn Experiment Runner

This script runs Anthropic Claude models on multiple datasets using deterministic sampling
with enhanced early termination logic for efficiency and cost control.

Features:
- Deterministic sampling for reproducible experiments
- Early termination if final_accuracy = 0.0 or refusal patterns detected
- Support for multiple Claude models (Haiku, Sonnet, Opus)
- Comprehensive logging and progress tracking
- Cost estimation and monitoring

Usage:
    python3 run_anthropic_multiturn_experiments.py --datasets gsm8k,mathbench --sample_size 100
"""

import json
import sys
import argparse
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, skip

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.loop.runner import run_dataset
from src.data.scaling_datasets import ScalingDatasetManager
from src.utils.enhanced_trace_formatter import EnhancedTraceFormatter
from src.eval.csv_formatter import ReasoningCSVFormatter

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AnthropicMultiTurnExperimentRunner:
    """Enhanced runner for Anthropic multi-turn experiments with early termination."""
    
    def __init__(self, output_dir: str = "outputs/anthropic_multiturn"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Anthropic model configurations
        self.anthropic_models = {
            "claude-3-haiku-20240307": {
                "name": "Claude-3-Haiku",
                "cost_per_1k_input": 0.00025,
                "cost_per_1k_output": 0.00125,
                "max_tokens": 4096
            },
            "claude-3-5-sonnet-20241022": {
                "name": "Claude-3.5-Sonnet", 
                "cost_per_1k_input": 0.003,
                "cost_per_1k_output": 0.015,
                "max_tokens": 8192
            },
            "claude-3-opus-20240229": {
                "name": "Claude-3-Opus",
                "cost_per_1k_input": 0.015,
                "cost_per_1k_output": 0.075,
                "max_tokens": 4096
            }
        }
        
        # Default experiment configuration
        self.default_config = {
            "max_turns": 3,
            "temperature": 0.2,
            "deterministic_sampling": True,
            "early_termination": True,
            "termination_conditions": {
                "zero_accuracy": True,
                "refusal_patterns": True,
                "max_consecutive_zeros": 5
            }
        }
        
        self.experiment_log = []
        self.total_cost = 0.0
        
    def detect_refusal_patterns(self, trace: Dict[str, Any]) -> bool:
        """
        Detect if the model is consistently refusing to answer questions.
        
        Args:
            trace: Experiment trace containing turns and answers
            
        Returns:
            True if refusal patterns detected
        """
        if not trace or not trace.get('turns'):
            return False
            
        # Check if all answers are exactly "0" or similar refusal indicators
        turns = trace.get('turns', [])
        if not turns:
            return False
            
        refusal_indicators = ["0", "n/a", "unknown", "cannot", "refuse", "decline", ""]
        
        # Check final answer
        final_answer = str(trace.get('final_answer', '')).lower().strip()
        if final_answer in refusal_indicators:
            logger.warning(f"Potential refusal detected - final_answer: '{final_answer}'")
            return True
            
        # Check if all turn answers are refusals
        turn_answers = []
        for turn in turns:
            answer = str(turn.get('answer', '')).lower().strip()
            turn_answers.append(answer)
            
        # If all answers are refusal indicators
        if all(answer in refusal_indicators for answer in turn_answers):
            logger.warning(f"All turn answers are refusals: {turn_answers}")
            return True
            
        return False
    
    def should_terminate_early(self, experiment_results: List[Dict[str, Any]], 
                              current_trace: Dict[str, Any] = None) -> Tuple[bool, str]:
        """
        Determine if experiment should terminate early based on conditions.
        
        Args:
            experiment_results: List of completed experiment traces
            current_trace: Current trace being processed (optional)
            
        Returns:
            (should_terminate, reason)
        """
        if not self.default_config.get('early_termination', True):
            return False, ""
            
        # Check termination conditions
        conditions = self.default_config.get('termination_conditions', {})
        
        # Condition 1: Check if current trace has zero accuracy
        if conditions.get('zero_accuracy', True) and current_trace:
            final_accuracy = current_trace.get('final_accuracy', 1)
            if final_accuracy == 0.0:
                logger.info(f"Early termination: final_accuracy = 0.0")
                return True, "final_accuracy_zero"
        
        # Condition 2: Check for refusal patterns in current trace
        if conditions.get('refusal_patterns', True) and current_trace:
            if self.detect_refusal_patterns(current_trace):
                logger.info(f"Early termination: refusal patterns detected")
                return True, "refusal_patterns"
        
        # Condition 3: Check for consecutive zero accuracies
        max_consecutive = conditions.get('max_consecutive_zeros', 5)
        if max_consecutive > 0 and len(experiment_results) >= max_consecutive:
            recent_accuracies = [
                trace.get('final_accuracy', 1) 
                for trace in experiment_results[-max_consecutive:]
            ]
            if all(acc == 0.0 for acc in recent_accuracies):
                logger.info(f"Early termination: {max_consecutive} consecutive zero accuracies")
                return True, f"consecutive_zeros_{max_consecutive}"
        
        return False, ""
    
    def load_deterministic_dataset(self, dataset_name: str, sample_size: int) -> List[Dict[str, Any]]:
        """Load dataset using deterministic sampling."""
        logger.info(f"Loading deterministic dataset: {dataset_name} with {sample_size} samples")
        
        # Try deterministic file first
        deterministic_file = Path(f"data/scaling/{dataset_name}_deterministic_{sample_size}.json")
        
        if deterministic_file.exists():
            logger.info(f"Using deterministic file: {deterministic_file}")
            with open(deterministic_file, 'r') as f:
                data = json.load(f)
            return data.get("samples", [])
        
        # Fallback to ScalingDatasetManager with fixed seed
        logger.info(f"Deterministic file not found, using seeded sampling")
        dm = ScalingDatasetManager()
        return dm.load_dataset(dataset_name, sample_size=sample_size, seed=42)
    
    def estimate_experiment_cost(self, model: str, dataset_size: int, max_turns: int = 3) -> float:
        """Estimate cost for an experiment."""
        if model not in self.anthropic_models:
            return 0.0
            
        model_config = self.anthropic_models[model]
        
        # Rough estimates (tokens per question/answer)
        avg_input_tokens_per_turn = 500  # Question + context + previous turns
        avg_output_tokens_per_turn = 200  # Answer + reasoning
        
        total_input_tokens = dataset_size * avg_input_tokens_per_turn * max_turns
        total_output_tokens = dataset_size * avg_output_tokens_per_turn * max_turns
        
        cost = (
            (total_input_tokens / 1000) * model_config["cost_per_1k_input"] +
            (total_output_tokens / 1000) * model_config["cost_per_1k_output"]
        )
        
        return cost
    
    def run_single_experiment(self, model: str, dataset_name: str, sample_size: int) -> Dict[str, Any]:
        """Run a single multi-turn experiment with early termination."""
        
        experiment_id = f"{model}_{dataset_name}_{sample_size}_{int(time.time())}"
        logger.info(f"Starting experiment: {experiment_id}")
        
        # Load deterministic dataset
        try:
            dataset = self.load_deterministic_dataset(dataset_name, sample_size)
            if not dataset:
                raise ValueError(f"No data loaded for {dataset_name} with {sample_size} samples")
                
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_name}: {e}")
            return {
                "experiment_id": experiment_id,
                "status": "failed",
                "error": f"Dataset loading failed: {e}",
                "model": model,
                "dataset": dataset_name,
                "sample_size": sample_size
            }
        
        # Estimate cost
        estimated_cost = self.estimate_experiment_cost(model, len(dataset), self.default_config["max_turns"])
        logger.info(f"Estimated cost: ${estimated_cost:.4f}")
        
        # Create temporary dataset file in the format expected by the loader
        temp_dataset_file = self.output_dir / f"{experiment_id}_dataset.json"
        with open(temp_dataset_file, 'w') as f:
            # For ToolQA and other datasets, the loader expects a direct list
            json.dump(dataset, f)
        
        experiment_result = {
            "experiment_id": experiment_id,
            "model": model,
            "dataset": dataset_name,
            "sample_size": sample_size,
            "actual_samples": len(dataset),
            "estimated_cost": estimated_cost,
            "start_time": datetime.now().isoformat(),
            "config": self.default_config.copy(),
            "early_termination_reason": None,
            "traces_processed": 0,
            "traces_with_zero_accuracy": 0,
            "traces_with_refusal": 0
        }
        
        try:
            # Run the experiment using the main pipeline
            traces_output = self.output_dir / f"{experiment_id}_traces.json"
            
            # Configure experiment to use deterministic sampling
            experiment_config = {
                "dataset_name": dataset_name,
                "model": model,
                "provider": "anthropic",
                "temperature": self.default_config["temperature"],
                "max_turns": self.default_config["max_turns"],
                "experiment_id": experiment_id,
                "deterministic_sampling": True,
                "features": {
                    "enable_confidence": True,
                    "enable_error_awareness": True,
                    "enable_multi_turn": True
                }
            }
            
            result = run_dataset(
                dataset_csv=str(temp_dataset_file),
                traces_out=str(traces_output),
                max_turns=self.default_config["max_turns"],
                provider="anthropic",
                model=model,
                config=experiment_config,
                experiment_id=experiment_id,
                dataset_name=dataset_name
            )
            
            # Process results and check for early termination conditions
            experiment_traces = []
            early_termination_reason = None
            
            # Load and analyze traces if they exist
            if traces_output.exists():
                try:
                    with open(traces_output, 'r') as f:
                        traces_data = json.load(f)
                    
                    if isinstance(traces_data, list):
                        experiment_traces = traces_data
                    elif isinstance(traces_data, dict) and 'traces' in traces_data:
                        experiment_traces = traces_data['traces']
                        
                except Exception as e:
                    logger.warning(f"Could not load traces for analysis: {e}")
            
            # Analyze traces for termination patterns
            zero_accuracy_count = 0
            refusal_count = 0
            
            for trace in experiment_traces:
                if trace.get('final_accuracy') == 0.0:
                    zero_accuracy_count += 1
                    
                if self.detect_refusal_patterns(trace):
                    refusal_count += 1
            
            # Check if we should have terminated early
            should_terminate, reason = self.should_terminate_early(experiment_traces)
            if should_terminate:
                early_termination_reason = reason
                logger.info(f"Experiment would benefit from early termination: {reason}")
            
            # Update experiment result
            experiment_result.update({
                "status": "completed",
                "end_time": datetime.now().isoformat(),
                "result": result,
                "traces_processed": len(experiment_traces),
                "traces_with_zero_accuracy": zero_accuracy_count,
                "traces_with_refusal": refusal_count,
                "early_termination_reason": early_termination_reason,
                "traces_output_file": str(traces_output)
            })
            
            # Calculate duration
            start_time = datetime.fromisoformat(experiment_result["start_time"])
            end_time = datetime.fromisoformat(experiment_result["end_time"])
            duration_seconds = (end_time - start_time).total_seconds()
            experiment_result["duration_seconds"] = duration_seconds
            
            logger.info(f"Experiment completed: {experiment_id}")
            logger.info(f"  Duration: {duration_seconds:.1f}s")
            logger.info(f"  Traces processed: {len(experiment_traces)}")
            logger.info(f"  Zero accuracy: {zero_accuracy_count}")
            logger.info(f"  Refusals detected: {refusal_count}")
            
        except Exception as e:
            logger.error(f"Experiment failed: {experiment_id} - {e}")
            experiment_result.update({
                "status": "failed",
                "error": str(e),
                "end_time": datetime.now().isoformat()
            })
        
        finally:
            # Clean up temporary dataset file
            if temp_dataset_file.exists():
                temp_dataset_file.unlink()
        
        return experiment_result
    
    def run_experiments(self, models: List[str], datasets: List[str], 
                       sample_sizes: List[int] = None) -> Dict[str, Any]:
        """Run multiple experiments with progress tracking."""
        
        if sample_sizes is None:
            sample_sizes = [100, 500]  # Default sample sizes
            
        # Filter to only Anthropic models
        valid_models = [m for m in models if m in self.anthropic_models]
        if not valid_models:
            raise ValueError(f"No valid Anthropic models found in: {models}")
            
        logger.info(f"Starting Anthropic multi-turn experiments")
        logger.info(f"Models: {valid_models}")
        logger.info(f"Datasets: {datasets}")  
        logger.info(f"Sample sizes: {sample_sizes}")
        
        # Calculate total experiments
        total_experiments = len(valid_models) * len(datasets) * len(sample_sizes)
        
        experiment_suite = {
            "suite_info": {
                "name": "Anthropic Multi-Turn with Early Termination",
                "start_time": datetime.now().isoformat(),
                "models": valid_models,
                "datasets": datasets,
                "sample_sizes": sample_sizes,
                "total_experiments": total_experiments,
                "config": self.default_config
            },
            "experiments": [],
            "summary": {}
        }
        
        experiment_count = 0
        successful_experiments = 0
        total_estimated_cost = 0.0
        
        # Run experiments
        for model in valid_models:
            for dataset in datasets:
                for sample_size in sample_sizes:
                    experiment_count += 1
                    
                    logger.info(f"\n{'='*60}")
                    logger.info(f"Experiment {experiment_count}/{total_experiments}")
                    logger.info(f"Model: {self.anthropic_models[model]['name']}")
                    logger.info(f"Dataset: {dataset}")
                    logger.info(f"Sample size: {sample_size}")
                    logger.info(f"{'='*60}")
                    
                    # Run experiment
                    result = self.run_single_experiment(model, dataset, sample_size)
                    experiment_suite["experiments"].append(result)
                    
                    # Track success and costs
                    if result["status"] == "completed":
                        successful_experiments += 1
                        
                    total_estimated_cost += result.get("estimated_cost", 0.0)
                    
                    # Log progress
                    logger.info(f"Progress: {experiment_count}/{total_experiments} "
                              f"({experiment_count/total_experiments*100:.1f}%)")
                    logger.info(f"Successful: {successful_experiments}/{experiment_count}")
                    
                    # Brief pause between experiments
                    time.sleep(2)
        
        # Calculate final summary
        experiment_suite["suite_info"]["end_time"] = datetime.now().isoformat()
        experiment_suite["summary"] = {
            "total_experiments": total_experiments,
            "successful_experiments": successful_experiments,
            "failed_experiments": total_experiments - successful_experiments,
            "success_rate": successful_experiments / total_experiments if total_experiments > 0 else 0,
            "total_estimated_cost": total_estimated_cost,
            "early_terminations": len([e for e in experiment_suite["experiments"] 
                                    if e.get("early_termination_reason")]),
            "zero_accuracy_experiments": len([e for e in experiment_suite["experiments"] 
                                            if e.get("traces_with_zero_accuracy", 0) > 0]),
            "refusal_experiments": len([e for e in experiment_suite["experiments"] 
                                     if e.get("traces_with_refusal", 0) > 0])
        }
        
        # Calculate duration
        start_time = datetime.fromisoformat(experiment_suite["suite_info"]["start_time"])
        end_time = datetime.fromisoformat(experiment_suite["suite_info"]["end_time"])
        duration_minutes = (end_time - start_time).total_seconds() / 60
        experiment_suite["summary"]["duration_minutes"] = round(duration_minutes, 2)
        
        # Save results
        results_file = self.output_dir / f"anthropic_multiturn_results_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump(experiment_suite, f, indent=2, default=str)
            
        logger.info(f"\n{'='*60}")
        logger.info("ANTHROPIC MULTI-TURN EXPERIMENTS COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Total experiments: {total_experiments}")
        logger.info(f"Successful: {successful_experiments}")
        logger.info(f"Success rate: {experiment_suite['summary']['success_rate']*100:.1f}%")
        logger.info(f"Duration: {duration_minutes:.1f} minutes")
        logger.info(f"Estimated cost: ${total_estimated_cost:.4f}")
        logger.info(f"Early terminations: {experiment_suite['summary']['early_terminations']}")
        logger.info(f"Results saved to: {results_file}")
        
        return experiment_suite

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run Anthropic multi-turn experiments")
    parser.add_argument("--models", 
                       default="claude-3-haiku-20240307,claude-3-5-sonnet-20241022",
                       help="Comma-separated list of Anthropic models")
    parser.add_argument("--datasets", 
                       default="gsm8k,mathbench",
                       help="Comma-separated list of datasets")
    parser.add_argument("--sample_sizes", 
                       default="100,500",
                       help="Comma-separated list of sample sizes")
    parser.add_argument("--output_dir", 
                       default="outputs/anthropic_multiturn",
                       help="Output directory for results")
    parser.add_argument("--max_turns", type=int, default=3,
                       help="Maximum number of turns")
    parser.add_argument("--temperature", type=float, default=0.2,
                       help="Temperature for model sampling")
    parser.add_argument("--disable_early_termination", action="store_true",
                       help="Disable early termination logic")
    
    args = parser.parse_args()
    
    # Parse lists
    models = [m.strip() for m in args.models.split(",")]
    datasets = [d.strip() for d in args.datasets.split(",")]
    sample_sizes = [int(s.strip()) for s in args.sample_sizes.split(",")]
    
    # Create runner
    runner = AnthropicMultiTurnExperimentRunner(args.output_dir)
    
    # Update configuration
    runner.default_config.update({
        "max_turns": args.max_turns,
        "temperature": args.temperature,
        "early_termination": not args.disable_early_termination
    })
    
    logger.info(f"Configuration: {runner.default_config}")
    
    # Run experiments
    try:
        results = runner.run_experiments(models, datasets, sample_sizes)
        print(f"\n✅ All experiments completed successfully!")
        print(f"Results saved to: {args.output_dir}")
        return 0
        
    except Exception as e:
        logger.error(f"Experiment suite failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())