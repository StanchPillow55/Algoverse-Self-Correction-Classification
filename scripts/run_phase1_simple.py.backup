#!/usr/bin/env python3
"""
Simple Phase 1 Validation Runner

Runs Phase 1 validation using existing datasets and the main pipeline.
"""

import sys
import json
import time
import subprocess
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.trace_formatter import TraceFormatter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_phase1_simple():
    """Run Phase 1 validation using existing datasets."""
    logger.info("🚀 Starting Phase 1 Validation (Simple)")
    logger.info("   2 models × 1 dataset × 100 samples")
    logger.info("   Models: gpt-4o-mini, claude-haiku")
    logger.info("   Dataset: gsm8k")
    
    # Create output directory
    output_dir = Path("outputs/phase1_simple")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize trace formatter
    trace_formatter = TraceFormatter(str(output_dir / "formatted_traces"))
    
    # Models to test (display names)
    models = [
        {"name": "gpt-4o-mini", "provider": "openai"},
        {"name": "claude-haiku", "provider": "anthropic"}
    ]

    # Map display names to API model IDs
    api_model_mapping = {
        "gpt-4o-mini": "gpt-4o-mini",
        "gpt-4o": "gpt-4o",
        "gpt-4": "gpt-4",
        "claude-haiku": "claude-3-haiku-20240307",
        "claude-sonnet": "claude-3-sonnet-20240229",
        "claude-opus": "claude-3-opus-20240229",
    }
    
    # Dataset
    dataset = "data/gsm8k/test_100.jsonl"
    
    results = []
    start_time = time.time()
    
    for model in models:
        logger.info(f"   Running {model['name']} on GSM8K...")
        
        # Create output directory for this model
        model_output_dir = output_dir / f"{model['name']}_gsm8k"
        model_output_dir.mkdir(exist_ok=True)
        
        # Resolve API model ID and run the experiment using the main pipeline
        api_model = api_model_mapping.get(model["name"], model["name"])
        cmd = [
            "python", "-m", "src.main", "run",
            "--dataset", dataset,
            "--out", str(model_output_dir / "traces.jsonl"),
            "--max-turns", "3",
            "--provider", model["provider"],
            "--model", api_model,
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logger.info(f"   ✅ {model['name']} completed successfully")
                
                # Format traces
                traces_file = model_output_dir / "traces.jsonl"
                if traces_file.exists():
                    experiment_id = f"phase1_{model['name']}_gsm8k"
                    formatted = trace_formatter.format_experiment_traces(
                        str(traces_file), experiment_id
                    )
                    
                    # Calculate basic metrics
                    with open(traces_file, 'r') as f:
                        traces = [json.loads(line) for line in f if line.strip()]
                    
                    total_samples = len(traces)
                    correct_samples = sum(1 for trace in traces if trace.get('final_correct', False))
                    accuracy = correct_samples / total_samples if total_samples > 0 else 0
                    
                    result_data = {
                        "model_name": model["name"],
                        "provider": model["provider"],
                        "dataset": "gsm8k",
                        "sample_size": total_samples,
                        "accuracy": accuracy,
                        "correct_samples": correct_samples,
                        "total_samples": total_samples,
                        "traces_file": str(traces_file),
                        "formatted_traces": formatted
                    }
                    results.append(result_data)
                    
                    logger.info(f"   📊 {model['name']}: {accuracy:.3f} accuracy ({correct_samples}/{total_samples})")
                else:
                    logger.error(f"   ❌ No traces file created for {model['name']}")
            else:
                logger.error(f"   ❌ {model['name']} failed: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            logger.error(f"   ❌ {model['name']} timed out")
        except Exception as e:
            logger.error(f"   ❌ {model['name']} error: {e}")
    
    end_time = time.time()
    duration = end_time - start_time
    
    logger.info(f"✅ Phase 1 completed in {duration:.1f}s")
    
    # Save results
    results_file = output_dir / "phase1_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            "phase": "phase1_validation_simple",
            "duration": duration,
            "experiments": results
        }, f, indent=2)
    
    logger.info(f"📊 Results saved to: {results_file}")
    
    # Print summary
    print("\n📊 Phase 1 Validation Summary")
    print("=" * 40)
    for result in results:
        print(f"  {result['model_name']} on {result['dataset']}: {result['accuracy']:.3f} accuracy ({result['correct_samples']}/{result['total_samples']})")
    
    return results

if __name__ == "__main__":
    run_phase1_simple()
