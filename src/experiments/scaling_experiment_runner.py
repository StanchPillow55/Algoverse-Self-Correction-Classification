"""
Scaling Experiment Runner

Runs self-correction experiments across multiple models and datasets
for the scaling study, with cost tracking and result aggregation.
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np

# Handle imports for both script and module usage
try:
    from ..utils.multi_model_manager import MultiModelManager, ModelConfig
    from ..loop.runner import run_dataset
    from ..utils.trace_logger import TraceLogger
except ImportError:
    from utils.multi_model_manager import MultiModelManager, ModelConfig
    from loop.runner import run_dataset
    from utils.trace_logger import TraceLogger

logger = logging.getLogger(__name__)

@dataclass
class ScalingExperimentConfig:
    """Configuration for scaling experiments."""
    models: List[str]  # Model names to test
    datasets: List[str]  # Dataset names to test
    sample_sizes: List[int]  # Sample sizes for each dataset
    max_turns: int = 3  # Maximum self-correction turns
    output_dir: str = "outputs/scaling_experiments"
    temperature: float = 0.0
    max_tokens: int = 4000

@dataclass
class ExperimentResult:
    """Result of a single experiment run."""
    model_name: str
    dataset_name: str
    sample_size: int
    initial_accuracy: float
    final_accuracy: float
    improvement: float
    total_tokens: int
    cost: float
    latency: float
    metadata: Dict[str, Any]

class ScalingExperimentRunner:
    """Runs scaling experiments across models and datasets."""
    
    def __init__(self, config: ScalingExperimentConfig):
        """Initialize the experiment runner."""
        self.config = config
        self.model_manager = MultiModelManager()
        self.trace_logger = TraceLogger()
        
        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Results storage
        self.results: List[ExperimentResult] = []
        
    def run_experiment(self, model_name: str, dataset_name: str, 
                      sample_size: int) -> ExperimentResult:
        """Run a single experiment using the existing pipeline."""
        logger.info(f"Running experiment: {model_name} on {dataset_name} (n={sample_size})")
        
        # Get model configuration
        model_config = self.model_manager.get_model_config(model_name)
        if not model_config:
            raise ValueError(f"Model {model_name} not found in configuration")
        
        # Determine provider from model config
        provider = model_config.provider
        
        # Create dataset path
        dataset_path = self._get_dataset_path(dataset_name, sample_size)
        if not dataset_path.exists():
            raise ValueError(f"Dataset not found: {dataset_path}")
        
        # Create output file
        output_file = self.output_dir / f"{model_name}_{dataset_name}_{sample_size}.json"
        
        # Set environment variables for the model
        os.environ["PROVIDER"] = provider
        os.environ["DEMO_MODE"] = "0"  # Use real API
        os.environ["OPENAI_TEMPERATURE"] = str(self.config.temperature)
        os.environ["OPENAI_MAX_TOKENS"] = str(self.config.max_tokens)
        
        # Set model-specific environment variables
        if provider == "openai":
            os.environ["OPENAI_MODEL"] = model_config.model_id
        elif provider == "anthropic":
            os.environ["ANTHROPIC_MODEL"] = model_config.model_id
        elif provider == "replicate":
            os.environ["REPLICATE_MODEL"] = model_config.model_id
        
        start_time = time.time()
        
        try:
            # Run using existing pipeline
            result = run_dataset(
                dataset_csv=str(dataset_path),
                traces_out=str(output_file),
                max_turns=self.config.max_turns,
                provider=provider,
                model=model_config.model_id
            )
            
            # Calculate metrics
            traces = result.get("traces", [])
            if not traces:
                raise ValueError("No traces generated")
            
            # Calculate initial and final accuracy
            initial_correct = sum(1 for t in traces if t.get("turns", [{}])[0].get("accuracy", 0) == 1)
            final_correct = sum(1 for t in traces if t.get("final_accuracy", 0) == 1)
            
            initial_accuracy = initial_correct / len(traces)
            final_accuracy = final_correct / len(traces)
            improvement = final_accuracy - initial_accuracy
            
            # Estimate cost and tokens
            total_tokens = self._estimate_tokens(traces)
            cost = (total_tokens / 1000) * model_config.estimated_cost_per_1k_tokens
            
            # Calculate latency
            latency = time.time() - start_time
            
            experiment_result = ExperimentResult(
                model_name=model_name,
                dataset_name=dataset_name,
                sample_size=sample_size,
                initial_accuracy=initial_accuracy,
                final_accuracy=final_accuracy,
                improvement=improvement,
                total_tokens=total_tokens,
                cost=cost,
                latency=latency,
                metadata={
                    "max_turns": self.config.max_turns,
                    "temperature": self.config.temperature,
                    "provider": provider,
                    "model_id": model_config.model_id,
                    "timestamp": time.time()
                }
            )
            
            self.results.append(experiment_result)
            return experiment_result
            
        except Exception as e:
            logger.error(f"Experiment failed: {model_name} on {dataset_name}: {e}")
            raise
    
    def _get_dataset_path(self, dataset_name: str, sample_size: int) -> Path:
        """Get the path to a dataset with specific sample size."""
        if dataset_name == "toolqa":
            return Path(f"data/scaling/toolqa_sample_{sample_size}.csv")
        elif dataset_name == "superglue":
            return Path(f"data/scaling/superglue_sample_{sample_size}.csv")
        elif dataset_name == "college_math":
            return Path(f"data/scaling/mathbench_sample_{sample_size}.csv")
        elif dataset_name == "humaneval":
            return Path("data/humaneval/HumanEval.jsonl")  # Use existing dataset
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    
    def _estimate_tokens(self, traces: List[Dict]) -> int:
        """Estimate total tokens used in the experiment."""
        total_tokens = 0
        for trace in traces:
            turns = trace.get("turns", [])
            for turn in turns:
                # Rough token estimation
                answer = turn.get("answer", "")
                total_tokens += len(answer.split()) * 1.3  # Rough token estimation
        return int(total_tokens)
    
    def run_phase_experiments(self, phase: str) -> List[ExperimentResult]:
        """Run experiments for a specific phase."""
        phase_config = self._get_phase_config(phase)
        if not phase_config:
            raise ValueError(f"Unknown phase: {phase}")
        
        logger.info(f"Running {phase} experiments")
        logger.info(f"Models: {phase_config['models']}")
        logger.info(f"Datasets: {phase_config['datasets']}")
        logger.info(f"Sample size: {phase_config['sample_size']}")
        
        phase_results = []
        
        for model_name in phase_config['models']:
            for dataset_name in phase_config['datasets']:
                try:
                    result = self.run_experiment(
                        model_name=model_name,
                        dataset_name=dataset_name,
                        sample_size=phase_config['sample_size']
                    )
                    phase_results.append(result)
                    logger.info(f"✓ {model_name} on {dataset_name}: {result.improvement:.3f} improvement")
                except Exception as e:
                    logger.error(f"✗ {model_name} on {dataset_name}: {e}")
                    continue
        
        return phase_results
    
    def _get_phase_config(self, phase: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific phase."""
        phase_configs = {
            "phase1": {
                "models": ["gpt-4o-mini", "claude-haiku"],
                "datasets": ["toolqa"],
                "sample_size": 100
            },
            "phase2": {
                "models": ["gpt-4o-mini", "claude-haiku", "gpt-4o", "claude-sonnet"],
                "datasets": ["toolqa", "superglue"],
                "sample_size": 500
            },
            "phase3": {
                "models": ["gpt-4o-mini", "claude-haiku", "gpt-4o", "claude-sonnet", "gpt-4", "claude-opus"],
                "datasets": ["toolqa", "superglue", "college_math", "humaneval"],
                "sample_size": 1000
            }
        }
        return phase_configs.get(phase)
    
    def generate_analysis_report(self) -> Dict[str, Any]:
        """Generate analysis report from all results."""
        if not self.results:
            return {"error": "No results to analyze"}
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame([
            {
                "model": r.model_name,
                "dataset": r.dataset_name,
                "sample_size": r.sample_size,
                "initial_accuracy": r.initial_accuracy,
                "final_accuracy": r.final_accuracy,
                "improvement": r.improvement,
                "cost": r.cost,
                "tokens": r.total_tokens
            }
            for r in self.results
        ])
        
        # Calculate scaling correlations
        analysis = {
            "summary": {
                "total_experiments": len(self.results),
                "total_cost": sum(r.cost for r in self.results),
                "total_tokens": sum(r.total_tokens for r in self.results)
            },
            "by_model": {},
            "by_dataset": {},
            "scaling_analysis": {}
        }
        
        # Analysis by model
        for model in df["model"].unique():
            model_data = df[df["model"] == model]
            analysis["by_model"][model] = {
                "avg_improvement": model_data["improvement"].mean(),
                "avg_cost": model_data["cost"].mean(),
                "experiments": len(model_data)
            }
        
        # Analysis by dataset
        for dataset in df["dataset"].unique():
            dataset_data = df[df["dataset"] == dataset]
            analysis["by_dataset"][dataset] = {
                "avg_improvement": dataset_data["improvement"].mean(),
                "avg_cost": dataset_data["cost"].mean(),
                "experiments": len(dataset_data)
            }
        
        # Scaling analysis
        analysis["scaling_analysis"] = {
            "cost_vs_improvement": df[["cost", "improvement"]].corr().iloc[0, 1],
            "sample_size_vs_improvement": df[["sample_size", "improvement"]].corr().iloc[0, 1]
        }
        
        # Save analysis
        analysis_path = self.output_dir / "analysis_report.json"
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        return analysis

def run_scaling_experiment(models: List[str], datasets: List[str], 
                          sample_sizes: List[int] = [100, 500, 1000]) -> ScalingExperimentRunner:
    """Convenience function to run a scaling experiment."""
    config = ScalingExperimentConfig(
        models=models,
        datasets=datasets,
        sample_sizes=sample_sizes
    )
    
    runner = ScalingExperimentRunner(config)
    return runner
