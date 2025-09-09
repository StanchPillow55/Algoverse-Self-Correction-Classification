#!/usr/bin/env python3
"""
Phase 1 Validation Runner

Runs Phase 1 validation experiments: 2 models × 1 dataset × 100 samples
"""

import sys
import json
import time
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.experiments.scaling_runner import ScalingExperimentRunner, ScalingExperimentConfig
from src.utils.trace_formatter import TraceFormatter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_phase1_validation():
    """Run Phase 1 validation experiments."""
    logger.info("🚀 Starting Phase 1 Validation")
    logger.info("   2 models × 1 dataset × 100 samples")
    logger.info("   Models: gpt-4o-mini, claude-haiku")
    logger.info("   Dataset: gsm8k")
    
    # Create output directory
    output_dir = Path("outputs/phase1_validation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize trace formatter
    trace_formatter = TraceFormatter(str(output_dir / "formatted_traces"))
    
    # Phase 1 configuration
    config = ScalingExperimentConfig(
        models=["gpt-4o-mini", "claude-haiku"],
        datasets=["gsm8k"],
        sample_sizes=[100],
        max_turns=3,
        temperature=0.2,
        output_dir=str(output_dir)
    )
    
    # Run experiments
    start_time = time.time()
    runner = ScalingExperimentRunner(config)
    results = runner.run_full_experiment()
    
    end_time = time.time()
    duration = end_time - start_time
    
    logger.info(f"✅ Phase 1 completed in {duration:.1f}s")
    
    # Convert results to dict format and format traces
    formatted_results = {}
    results_dict = []
    
    for result in results:
        # Convert ExperimentResult to dict
        result_dict = {
            "model_name": result.model_name,
            "dataset_name": result.dataset_name,
            "sample_size": result.sample_size,
            "initial_accuracy": result.initial_accuracy,
            "final_accuracy": result.final_accuracy,
            "improvement": result.improvement,
            "total_tokens": result.total_tokens,
            "cost": result.cost,
            "latency": result.latency,
            "metadata": result.metadata
        }
        results_dict.append(result_dict)
        
        # Look for traces file in the output directory
        traces_file = output_dir / f"{result.model_name}_{result.dataset_name}_{result.sample_size}_traces.json"
        if traces_file.exists():
            experiment_id = f"phase1_{result.model_name}_{result.dataset_name}"
            logger.info(f"   Formatting traces for {experiment_id}")
            formatted = trace_formatter.format_experiment_traces(
                str(traces_file), experiment_id
            )
            formatted_results[experiment_id] = formatted
            result_dict['formatted_traces'] = formatted
    
    # Save results
    results_file = output_dir / "phase1_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            "phase": "phase1_validation",
            "duration": duration,
            "experiments": results_dict,
            "formatted_traces": formatted_results
        }, f, indent=2)
    
    logger.info(f"📊 Results saved to: {results_file}")
    
    # Print summary
    print("\n📊 Phase 1 Validation Summary")
    print("=" * 40)
    for result in results:
        print(f"  {result.model_name} on {result.dataset_name}: {result.final_accuracy:.3f} accuracy, ${result.cost:.4f} cost")
    
    return results

if __name__ == "__main__":
    run_phase1_validation()
