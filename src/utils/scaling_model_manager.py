"""
Scaling Model Manager for Multi-Model Self-Correction Study

Manages multiple model providers for scaling law analysis across different model sizes.
Tracks costs and provides unified interface for model switching.
"""

import os
import json
import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging

# Handle imports for both script and module usage
try:
    from .config import Config
except ImportError:
    try:
        from src.utils.config import Config
    except ImportError:
        from config import Config

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for a model in the scaling study."""
    name: str
    provider: str
    model_id: str
    size_category: str  # "small", "medium", "large"
    estimated_cost_per_1k_tokens: float
    max_tokens: int = 4000
    temperature: float = 0.0
    available: bool = True
    description: str = ""

@dataclass
class ExperimentConfig:
    """Configuration for scaling experiments."""
    model_configs: List[ModelConfig]
    datasets: List[str]
    max_turns: int = 3
    sample_sizes: List[int] = None  # None means use full dataset
    
    def __post_init__(self):
        if self.sample_sizes is None:
            self.sample_sizes = [100, 500, 1000]  # Default sample sizes

class ScalingModelManager:
    """Manages multiple models for scaling study experiments."""
    
    def __init__(self, config_path: str = "configs/scaling_models.json"):
        """Initialize with model configurations."""
        self.config_path = Path(config_path)
        self.model_configs = self._load_model_configs()
        self.cost_tracker = CostTracker()
        
    def _load_model_configs(self) -> List[ModelConfig]:
        """Load model configurations from file or create defaults."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                data = json.load(f)
                return [ModelConfig(**config) for config in data.get('models', [])]
        else:
            return self._create_default_configs()
    
    def _create_default_configs(self) -> List[ModelConfig]:
        """Create default model configurations for the scaling study."""
        return [
            # Small Models (1-7B parameters)
            ModelConfig(
                name="gpt-4o-mini",
                provider="openai",
                model_id="gpt-4o-mini",
                size_category="small",
                estimated_cost_per_1k_tokens=0.00015
            ),
            ModelConfig(
                name="claude-haiku",
                provider="anthropic", 
                model_id="claude-3-haiku-20240307",
                size_category="small",
                estimated_cost_per_1k_tokens=0.00025
            ),
            
            # Medium Models (8-70B parameters)
            ModelConfig(
                name="gpt-4o",
                provider="openai",
                model_id="gpt-4o",
                size_category="medium", 
                estimated_cost_per_1k_tokens=0.0025
            ),
            ModelConfig(
                name="claude-sonnet",
                provider="anthropic",
                model_id="claude-3-5-sonnet-20241022", 
                size_category="medium",
                estimated_cost_per_1k_tokens=0.003
            ),
            ModelConfig(
                name="llama-70b",
                provider="replicate",
                model_id="meta-llama/llama-2-70b-chat",
                size_category="medium",
                estimated_cost_per_1k_tokens=0.0007
            ),
            
            # Large Models (100B+ parameters)
            ModelConfig(
                name="gpt-4",
                provider="openai",
                model_id="gpt-4",
                size_category="large",
                estimated_cost_per_1k_tokens=0.03
            ),
            ModelConfig(
                name="claude-opus",
                provider="anthropic",
                model_id="claude-3-opus-20240229",
                size_category="large", 
                estimated_cost_per_1k_tokens=0.015
            )
        ]
    
    def get_models_by_category(self, category: str) -> List[ModelConfig]:
        """Get models filtered by size category."""
        return [m for m in self.model_configs if m.size_category == category]
    
    def get_available_models(self) -> List[ModelConfig]:
        """Get all available models (with API keys)."""
        available = []
        for model in self.model_configs:
            if self._check_model_availability(model):
                available.append(model)
        return available
    
    def _check_model_availability(self, model: ModelConfig) -> bool:
        """Check if a model is available (has API key)."""
        api_keys = {
            "openai": os.getenv("OPENAI_API_KEY"),
            "anthropic": os.getenv("ANTHROPIC_API_KEY"),
            "replicate": os.getenv("REPLICATE_API_TOKEN")
        }
        return api_keys.get(model.provider) is not None
    
    def estimate_experiment_cost(self, dataset_sizes: List[int], 
                                models: List[str] = None) -> Dict[str, float]:
        """Estimate total cost for the scaling experiment."""
        if models is None:
            models = [m.name for m in self.get_available_models()]
        
        total_cost = 0.0
        cost_breakdown = {}
        
        # Estimate tokens per sample (conservative estimate)
        tokens_per_sample = 200  # Input + output tokens per sample
        turns_per_sample = 3     # Average self-correction turns
        
        for model_name in models:
            model = next((m for m in self.model_configs if m.name == model_name), None)
            if not model:
                continue
                
            model_cost = 0.0
            for dataset_size in dataset_sizes:
                total_tokens = dataset_size * tokens_per_sample * turns_per_sample
                cost = (total_tokens / 1000) * model.estimated_cost_per_1k_tokens
                model_cost += cost
                
            cost_breakdown[model_name] = model_cost
            total_cost += model_cost
        
        cost_breakdown["total"] = total_cost
        return cost_breakdown
    
    def get_model_client(self, model_name: str):
        """Get the appropriate client for a model."""
        model = next((m for m in self.model_configs if m.name == model_name), None)
        if not model:
            raise ValueError(f"Model {model_name} not found")
        
        if model.provider == "openai":
            from openai import OpenAI
            return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif model.provider == "anthropic":
            import anthropic
            return anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        elif model.provider == "replicate":
            import replicate
            return replicate
        else:
            raise ValueError(f"Unsupported provider: {model.provider}")
    
    def call_model(self, model_name: str, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        """Make a call to the specified model with cost tracking."""
        model = next((m for m in self.model_configs if m.name == model_name), None)
        if not model:
            raise ValueError(f"Model {model_name} not found")
        
        start_time = time.time()
        
        try:
            if model.provider == "openai":
                client = self.get_model_client(model_name)
                response = client.chat.completions.create(
                    model=model.model_id,
                    messages=messages,
                    temperature=kwargs.get('temperature', model.temperature),
                    max_tokens=kwargs.get('max_tokens', model.max_tokens)
                )
                content = response.choices[0].message.content
                usage = response.usage
                
            elif model.provider == "anthropic":
                client = self.get_model_client(model_name)
                response = client.messages.create(
                    model=model.model_id,
                    messages=messages,
                    temperature=kwargs.get('temperature', model.temperature),
                    max_tokens=kwargs.get('max_tokens', model.max_tokens)
                )
                content = response.content[0].text
                usage = response.usage
                
            else:
                raise ValueError(f"Provider {model.provider} not implemented yet")
            
            # Track costs
            self.cost_tracker.record_usage(
                model_name=model_name,
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
                cost_per_1k=model.estimated_cost_per_1k_tokens
            )
            
            return {
                "content": content,
                "usage": usage,
                "model": model_name,
                "latency": time.time() - start_time
            }
            
        except Exception as e:
            logger.error(f"Error calling model {model_name}: {e}")
            raise
    
    def save_config(self):
        """Save current model configurations to file."""
        config_data = {
            "models": [
                {
                    "name": m.name,
                    "provider": m.provider,
                    "model_id": m.model_id,
                    "size_category": m.size_category,
                    "estimated_cost_per_1k_tokens": m.estimated_cost_per_1k_tokens,
                    "max_tokens": m.max_tokens,
                    "temperature": m.temperature,
                    "available": m.available
                }
                for m in self.model_configs
            ]
        }
        
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """Get cost summary for all experiments."""
        return self.cost_tracker.get_summary()

class CostTracker:
    """Tracks costs across experiments."""
    
    def __init__(self):
        self.usage_records = []
        self.total_cost = 0.0
    
    def record_usage(self, model_name: str, input_tokens: int, 
                    output_tokens: int, cost_per_1k: float):
        """Record token usage and calculate cost."""
        total_tokens = input_tokens + output_tokens
        cost = (total_tokens / 1000) * cost_per_1k
        
        record = {
            "model": model_name,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "cost": cost,
            "timestamp": time.time()
        }
        
        self.usage_records.append(record)
        self.total_cost += cost
    
    def get_summary(self) -> Dict[str, Any]:
        """Get cost summary by model."""
        summary = {"total_cost": self.total_cost, "by_model": {}}
        
        for record in self.usage_records:
            model = record["model"]
            if model not in summary["by_model"]:
                summary["by_model"][model] = {
                    "total_cost": 0.0,
                    "total_tokens": 0,
                    "calls": 0
                }
            
            summary["by_model"][model]["total_cost"] += record["cost"]
            summary["by_model"][model]["total_tokens"] += record["total_tokens"]
            summary["by_model"][model]["calls"] += 1
        
        return summary

def get_scaling_manager() -> ScalingModelManager:
    """Get singleton instance of scaling model manager."""
    if not hasattr(get_scaling_manager, "_instance"):
        get_scaling_manager._instance = ScalingModelManager()
    return get_scaling_manager._instance
