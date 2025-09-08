"""
Scaling Experiment Runner

Runs self-correction experiments across multiple models and datasets
to analyze scaling laws and model performance correlations.
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import pandas as pd

from ..utils.scaling_model_manager import ScalingModelManager, ModelConfig
from ..data.scaling_datasets import ScalingDatasetManager
from ..utils.tracing import TraceLogger

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
        self.model_manager = ScalingModelManager()
        self.dataset_manager = ScalingDatasetManager()
        self.trace_logger = TraceLogger()
        
        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Results storage
        self.results: List[ExperimentResult] = []
        
    def run_experiment(self, model_name: str, dataset_name: str, 
                      sample_size: int) -> ExperimentResult:
        """Run a single experiment."""
        logger.info(f"Running experiment: {model_name} on {dataset_name} (n={sample_size})")
        
        # Load dataset
        samples = self.dataset_manager.load_dataset(dataset_name, sample_size)
        if not samples:
            raise ValueError(f"Failed to load dataset {dataset_name}")
        
        # Initialize metrics
        correct_initial = 0
        correct_final = 0
        total_tokens = 0
        total_cost = 0.0
        total_latency = 0.0
        
        # Process each sample
        for i, sample in enumerate(samples):
            logger.info(f"Processing sample {i+1}/{len(samples)}")
            
            try:
                # Get initial answer
                initial_result = self._get_initial_answer(model_name, sample)
                total_tokens += initial_result["usage"].total_tokens
                total_cost += self._calculate_cost(model_name, initial_result["usage"])
                total_latency += initial_result["latency"]
                
                # Check if initial answer is correct
                is_correct_initial = self._evaluate_answer(
                    initial_result["content"], sample, dataset_name
                )
                if is_correct_initial:
                    correct_initial += 1
                
                # Self-correction loop
                current_answer = initial_result["content"]
                current_correct = is_correct_initial
                
                for turn in range(self.config.max_turns):
                    if current_correct:
                        break  # Stop if already correct
                    
                    # Generate correction
                    correction_result = self._get_correction(
                        model_name, sample, current_answer, turn + 1
                    )
                    total_tokens += correction_result["usage"].total_tokens
                    total_cost += self._calculate_cost(model_name, correction_result["usage"])
                    total_latency += correction_result["latency"]
                    
                    # Update answer
                    current_answer = correction_result["content"]
                    current_correct = self._evaluate_answer(
                        current_answer, sample, dataset_name
                    )
                
                if current_correct:
                    correct_final += 1
                
            except Exception as e:
                logger.error(f"Error processing sample {i}: {e}")
                continue
        
        # Calculate final metrics
        initial_accuracy = correct_initial / len(samples)
        final_accuracy = correct_final / len(samples)
        improvement = final_accuracy - initial_accuracy
        
        result = ExperimentResult(
            model_name=model_name,
            dataset_name=dataset_name,
            sample_size=sample_size,
            initial_accuracy=initial_accuracy,
            final_accuracy=final_accuracy,
            improvement=improvement,
            total_tokens=total_tokens,
            cost=total_cost,
            latency=total_latency,
            metadata={
                "max_turns": self.config.max_turns,
                "temperature": self.config.temperature,
                "timestamp": time.time()
            }
        )
        
        self.results.append(result)
        return result
    
    def _get_initial_answer(self, model_name: str, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Get initial answer from the model."""
        question = sample.get("question", "")
        
        messages = [
            {"role": "system", "content": "Answer the question concisely and accurately."},
            {"role": "user", "content": question}
        ]
        
        return self.model_manager.call_model(
            model_name, 
            messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )
    
    def _get_correction(self, model_name: str, sample: Dict[str, Any], 
                       current_answer: str, turn: int) -> Dict[str, Any]:
        """Get self-correction from the model."""
        question = sample.get("question", "")
        correct_answer = sample.get("answer", "")
        
        correction_prompt = f"""
You previously answered: "{current_answer}"

The correct answer is: "{correct_answer}"

Please provide a corrected answer. Explain what was wrong with your previous answer and how you arrived at the correct one.
"""
        
        messages = [
            {"role": "system", "content": "You are helping to correct a previous answer. Be precise and explain your reasoning."},
            {"role": "user", "content": correction_prompt}
        ]
        
        return self.model_manager.call_model(
            model_name,
            messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )
    
    def _evaluate_answer(self, answer: str, sample: Dict[str, Any], 
                        dataset_name: str) -> bool:
        """Evaluate if an answer is correct."""
        correct_answer = sample.get("answer", "")
        
        # Simple string matching (can be enhanced with more sophisticated evaluation)
        answer_clean = answer.strip().lower()
        correct_clean = correct_answer.strip().lower()
        
        # Check for exact match
        if answer_clean == correct_clean:
            return True
        
        # Check if correct answer is contained in the response
        if correct_clean in answer_clean:
            return True
        
        # For math problems, try to extract numbers
        if dataset_name in ["mathbench", "gsm8k"]:
            import re
            answer_numbers = re.findall(r'-?\d+\.?\d*', answer_clean)
            correct_numbers = re.findall(r'-?\d+\.?\d*', correct_clean)
            
            if answer_numbers and correct_numbers:
                try:
                    answer_num = float(answer_numbers[0])
                    correct_num = float(correct_numbers[0])
                    return abs(answer_num - correct_num) < 1e-6
                except ValueError:
                    pass
        
        return False
    
    def _calculate_cost(self, model_name: str, usage) -> float:
        """Calculate cost for token usage."""
        model = next((m for m in self.model_manager.model_configs if m.name == model_name), None)
        if not model:
            return 0.0
        
        total_tokens = usage.prompt_tokens + usage.completion_tokens
        return (total_tokens / 1000) * model.estimated_cost_per_1k_tokens
    
    def run_full_experiment(self) -> List[ExperimentResult]:
        """Run all experiments in the configuration."""
        logger.info(f"Starting full scaling experiment with {len(self.config.models)} models and {len(self.config.datasets)} datasets")
        
        total_experiments = len(self.config.models) * len(self.config.datasets) * len(self.config.sample_sizes)
        current_experiment = 0
        
        for model_name in self.config.models:
            for dataset_name in self.config.datasets:
                for sample_size in self.config.sample_sizes:
                    current_experiment += 1
                    logger.info(f"Progress: {current_experiment}/{total_experiments}")
                    
                    try:
                        result = self.run_experiment(model_name, dataset_name, sample_size)
                        self._save_result(result)
                    except Exception as e:
                        logger.error(f"Failed experiment {model_name}-{dataset_name}-{sample_size}: {e}")
                        continue
        
        logger.info("Full scaling experiment completed")
        return self.results
    
    def _save_result(self, result: ExperimentResult):
        """Save individual result to file."""
        filename = f"{result.model_name}_{result.dataset_name}_{result.sample_size}.json"
        filepath = self.output_dir / filename
        
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
        
        with open(filepath, 'w') as f:
            json.dump(result_dict, f, indent=2)
    
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
    runner.run_full_experiment()
    return runner
