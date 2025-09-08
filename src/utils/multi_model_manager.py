"""
Multi-Model Manager for Scaling Study

Handles multiple model providers (OpenAI, Claude, Replicate) with unified interface
and cost tracking for the scaling study experiments.
"""

import os
import json
import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging

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

class MultiModelManager:
    """Manages multiple models for scaling study experiments."""
    
    def __init__(self, config_path: str = "configs/scaling_models.json"):
        """Initialize with model configurations."""
        self.config_path = Path(config_path)
        self.model_configs = self._load_model_configs()
        self.cost_tracker = CostTracker()
        
    def _load_model_configs(self) -> List[ModelConfig]:
        """Load model configurations from file."""
        if not self.config_path.exists():
            logger.error(f"Model config file not found: {self.config_path}")
            return []
        
        with open(self.config_path, 'r') as f:
            data = json.load(f)
        
        models = []
        for model_data in data.get('models', []):
            model = ModelConfig(
                name=model_data['name'],
                provider=model_data['provider'],
                model_id=model_data['model_id'],
                size_category=model_data['size_category'],
                estimated_cost_per_1k_tokens=model_data['estimated_cost_per_1k_tokens'],
                max_tokens=model_data.get('max_tokens', 4000),
                temperature=model_data.get('temperature', 0.0),
                available=model_data.get('available', True),
                description=model_data.get('description', '')
            )
            models.append(model)
        
        return models
    
    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """Get configuration for a specific model."""
        for model in self.model_configs:
            if model.name == model_name:
                return model
        return None
    
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
        if not model.available:
            return False
        
        api_keys = {
            "openai": os.getenv("OPENAI_API_KEY"),
            "anthropic": os.getenv("ANTHROPIC_API_KEY"),
            "replicate": os.getenv("REPLICATE_API_TOKEN")
        }
        return api_keys.get(model.provider) is not None
    
    def create_learner_bot(self, model_name: str) -> 'LearnerBot':
        """Create a LearnerBot instance for the specified model."""
        model_config = self.get_model_config(model_name)
        if not model_config:
            raise ValueError(f"Model {model_name} not found in configuration")
        
        if not self._check_model_availability(model_config):
            raise ValueError(f"Model {model_name} is not available (missing API key)")
        
        return LearnerBot(provider=model_config.provider, model=model_config.model_id)
    
    def call_model(self, model_name: str, prompt: str, history: List[Dict] = None, template: str = None) -> Tuple[str, float, Dict[str, Any]]:
        """Make a call to the specified model with cost tracking."""
        model_config = self.get_model_config(model_name)
        if not model_config:
            raise ValueError(f"Model {model_name} not found")
        
        if not self._check_model_availability(model_config):
            raise ValueError(f"Model {model_name} is not available")
        
        start_time = time.time()
        
        try:
            # Create learner bot for this model
            learner = self.create_learner_bot(model_name)
            
            # Make the call
            answer, confidence = learner.answer(prompt, history or [], template)
            
            # Estimate token usage (rough approximation)
            input_tokens = len(prompt.split()) * 1.3  # Rough token estimation
            output_tokens = len(answer.split()) * 1.3
            
            # Track costs
            self.cost_tracker.record_usage(
                model_name=model_name,
                input_tokens=int(input_tokens),
                output_tokens=int(output_tokens),
                cost_per_1k=model_config.estimated_cost_per_1k_tokens
            )
            
            return answer, confidence, {
                "model": model_name,
                "provider": model_config.provider,
                "latency": time.time() - start_time,
                "input_tokens": int(input_tokens),
                "output_tokens": int(output_tokens),
                "cost": (input_tokens + output_tokens) / 1000 * model_config.estimated_cost_per_1k_tokens
            }
            
        except Exception as e:
            logger.error(f"Error calling model {model_name}: {e}")
            raise
    
    def estimate_experiment_cost(self, models: List[str], datasets: List[str], sample_sizes: List[int]) -> Dict[str, Any]:
        """Estimate total cost for scaling experiments."""
        total_cost = 0.0
        breakdown = {}
        
        # Estimate tokens per sample (conservative estimate)
        tokens_per_sample = 200  # Input + output tokens per sample
        turns_per_sample = 3     # Average self-correction turns
        
        for model_name in models:
            model_config = self.get_model_config(model_name)
            if not model_config:
                continue
            
            model_cost = 0.0
            for dataset in datasets:
                for sample_size in sample_sizes:
                    total_tokens = sample_size * tokens_per_sample * turns_per_sample
                    cost = (total_tokens / 1000) * model_config.estimated_cost_per_1k_tokens
                    model_cost += cost
            
            breakdown[model_name] = model_cost
            total_cost += model_cost
        
        return {
            "total_cost": total_cost,
            "breakdown": breakdown,
            "models": models,
            "datasets": datasets,
            "sample_sizes": sample_sizes
        }
    
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

# Import LearnerBot here to avoid circular imports
try:
    from ..agents.learner import LearnerBot
except ImportError:
    from agents.learner import LearnerBot

def get_multi_model_manager() -> MultiModelManager:
    """Get singleton instance of multi-model manager."""
    if not hasattr(get_multi_model_manager, "_instance"):
        get_multi_model_manager._instance = MultiModelManager()
    return get_multi_model_manager._instance
