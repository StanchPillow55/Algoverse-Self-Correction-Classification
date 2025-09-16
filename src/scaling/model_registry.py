#!/usr/bin/env python3
"""
Model Registry for Scaling Laws Study

Provides standardized model configurations with parameter counts,
API costs, and provider details for the 7 target models in the scaling study.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import os


@dataclass
class ModelConfig:
    """Configuration for a model in the scaling study."""
    name: str
    provider: str
    parameter_count_b: float  # Billions of parameters
    cost_per_1k_tokens: float  # USD per 1K tokens
    size_category: str  # "Small", "Medium", "Large"
    api_model_name: str  # Actual API model identifier
    max_tokens: int = 4096
    supports_system_prompt: bool = True
    notes: str = ""


# Model Registry for Scaling Study
MODEL_REGISTRY: Dict[str, ModelConfig] = {
    # Small Models (1-7B parameters)
    "gpt-4o-mini": ModelConfig(
        name="GPT-4o-mini",
        provider="openai",
        parameter_count_b=1.8,
        cost_per_1k_tokens=0.00015,
        size_category="Small",
        api_model_name="gpt-4o-mini",
        notes="Efficient small model for cost-conscious applications"
    ),
    
    "claude-haiku": ModelConfig(
        name="Claude Haiku",
        provider="anthropic", 
        parameter_count_b=3.0,
        cost_per_1k_tokens=0.00025,
        size_category="Small",
        api_model_name="claude-3-haiku-20240307",
        notes="Fast and economical model from Anthropic"
    ),
    
    # Medium Models (8-70B parameters)
    "gpt-4o": ModelConfig(
        name="GPT-4o",
        provider="openai",
        parameter_count_b=8.0,  # Estimated
        cost_per_1k_tokens=0.0025,
        size_category="Medium",
        api_model_name="gpt-4o",
        notes="Balanced performance and cost model"
    ),
    
    "claude-sonnet": ModelConfig(
        name="Claude Sonnet",
        provider="anthropic",
        parameter_count_b=70.0,  # Estimated
        cost_per_1k_tokens=0.003,
        size_category="Medium", 
        api_model_name="claude-3-5-sonnet-20241022",
        notes="Mid-tier Anthropic model with strong reasoning (Claude 3.5)"
    ),
    
    "llama-70b": ModelConfig(
        name="Llama-70B",
        provider="replicate",
        parameter_count_b=70.0,
        cost_per_1k_tokens=0.0007,
        size_category="Medium",
        api_model_name="meta/llama-2-70b-chat",
        notes="Open-source large model via Replicate"
    ),
    
    # Large Models (100B+ parameters)
    "gpt-4": ModelConfig(
        name="GPT-4",
        provider="openai",
        parameter_count_b=175.0,  # Estimated
        cost_per_1k_tokens=0.03,
        size_category="Large",
        api_model_name="gpt-4",
        notes="High-capability model with premium pricing"
    ),
    
    "claude-opus": ModelConfig(
        name="Claude Opus", 
        provider="anthropic",
        parameter_count_b=175.0,  # Estimated
        cost_per_1k_tokens=0.015,
        size_category="Large",
        api_model_name="claude-3-opus-20240229",
        notes="Anthropic's most capable model"
    ),
    
    # Add explicit Claude 3.5 Sonnet entry for clarity
    "claude-3.5-sonnet": ModelConfig(
        name="Claude 3.5 Sonnet",
        provider="anthropic",
        parameter_count_b=70.0,  # Estimated
        cost_per_1k_tokens=0.003,
        size_category="Medium",
        api_model_name="claude-3-5-sonnet-20241022",
        notes="Latest Claude 3.5 Sonnet model (Oct 2024)"
    )
}


def get_model_config(model_key: str) -> Optional[ModelConfig]:
    """Get configuration for a model by key."""
    return MODEL_REGISTRY.get(model_key.lower())


def get_models_by_category(category: str) -> List[ModelConfig]:
    """Get all models in a size category."""
    return [config for config in MODEL_REGISTRY.values() 
            if config.size_category == category]


def get_all_model_keys() -> List[str]:
    """Get all available model keys."""
    return list(MODEL_REGISTRY.keys())


def get_scaling_study_models() -> List[str]:
    """Get the 7 models for the scaling study in order of size."""
    models = list(MODEL_REGISTRY.values())
    models.sort(key=lambda m: m.parameter_count_b)
    return [model.name.lower().replace(" ", "-").replace("-", "_") for model in models]


def estimate_experiment_cost(
    model_key: str,
    num_samples: int,
    avg_tokens_per_sample: int = 2000,
    num_runs: int = 3
) -> Dict[str, float]:
    """Estimate total cost for running an experiment."""
    config = get_model_config(model_key)
    if not config:
        raise ValueError(f"Unknown model: {model_key}")
    
    total_tokens = num_samples * avg_tokens_per_sample * num_runs
    total_cost = (total_tokens / 1000) * config.cost_per_1k_tokens
    
    return {
        "total_tokens": total_tokens,
        "total_cost_usd": total_cost,
        "cost_per_sample": total_cost / (num_samples * num_runs),
        "model": config.name,
        "parameter_count_b": config.parameter_count_b
    }


def get_cost_benefit_threshold() -> float:
    """Return the identified cost-benefit threshold in billions of parameters."""
    return 7.0  # Based on research findings


def print_model_summary():
    """Print a summary of all registered models."""
    print("Scaling Study Model Registry")
    print("=" * 50)
    
    for category in ["Small", "Medium", "Large"]:
        models = get_models_by_category(category)
        print(f"\n{category} Models ({len(models)}):")
        for model in models:
            print(f"  {model.name:<15} {model.parameter_count_b:>6.1f}B params  ${model.cost_per_1k_tokens:.5f}/1K tokens")


if __name__ == "__main__":
    print_model_summary()
    
    # Example cost estimation
    print("\n" + "=" * 50)
    print("Sample Cost Estimation (100 samples, 3 runs):")
    for model_key in ["gpt-4o-mini", "claude-sonnet", "gpt-4"]:
        cost_info = estimate_experiment_cost(model_key, 100, 2000, 3)
        print(f"{cost_info['model']:<15} ${cost_info['total_cost_usd']:>8.2f} total  ${cost_info['cost_per_sample']:>8.5f}/sample")