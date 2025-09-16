"""
Model manager for handling local models and API placeholders.

Provides easy swapping between local HuggingFace models and external APIs.
"""

from typing import Dict, Any, Optional, List
from ..utils.config import Config
import logging

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages model selection and API key handling."""
    
    def __init__(self):
        """Initialize the model manager."""
        Config.load()
        self.local_models = {
            "embedding": {
                "miniLM": "sentence-transformers/all-MiniLM-L6-v2",
                "mpnet": "sentence-transformers/all-mpnet-base-v2",
                "distilbert": "sentence-transformers/distilbert-base-nli-mean-tokens"
            }
        }
        self.api_models = {
            "embedding": {
                "openai": "text-embedding-ada-002",
                "openai_3": "text-embedding-3-small"
            },
            "generation": {
                "openai": "gpt-4",
                "anthropic": "claude-3-sonnet-20240229"
            }
        }
        
    def get_available_models(self) -> Dict[str, Dict[str, str]]:
        """Get all available models (local and API)."""
        return {
            "local": self.local_models,
            "api": self.api_models
        }
    
    def is_api_available(self, provider: str) -> bool:
        """Check if API keys are available for a provider."""
        api_keys = {
            "openai": Config.OPENAI_API_KEY,
            "anthropic": Config.ANTHROPIC_API_KEY
        }
        
        return api_keys.get(provider) is not None
    
    def get_recommended_embedding_model(self) -> str:
        """Get the recommended embedding model based on availability."""
        # Prefer API models if keys are available
        if self.is_api_available("openai"):
            return self.api_models["embedding"]["openai"]
        
        # Fall back to best local model
        return self.local_models["embedding"]["mpnet"]
    
    def get_recommended_generation_model(self) -> str:
        """Get the recommended generation model based on availability."""
        # Prefer API models if keys are available
        if self.is_api_available("openai"):
            return self.api_models["generation"]["openai"]
        elif self.is_api_available("anthropic"):
            return self.api_models["generation"]["anthropic"]
        
        # Return placeholder for local generation (not implemented yet)
        return "local_placeholder"
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get configuration for a specific model."""
        config = {
            "name": model_name,
            "type": "local" if not self._is_api_model(model_name) else "api",
            "available": True
        }
        
        if self._is_api_model(model_name):
            provider = self._get_api_provider(model_name)
            config["available"] = self.is_api_available(provider)
            config["provider"] = provider
            
        return config
    
    def _is_api_model(self, model_name: str) -> bool:
        """Check if a model is an API model."""
        for category in self.api_models.values():
            if model_name in category.values():
                return True
        return False
    
    def _get_api_provider(self, model_name: str) -> Optional[str]:
        """Get the provider for an API model."""
        for category in self.api_models.values():
            for provider, model in category.items():
                if model == model_name:
                    return provider.split("_")[0]  # Handle cases like "openai_3"
        return None
    
    def log_model_status(self):
        """Log the status of available models."""
        logger.info("Model availability status:")
        
        # Check API availability
        for provider in ["openai", "anthropic"]:
            status = "✓ Available" if self.is_api_available(provider) else "✗ No API key"
            logger.info(f"  {provider.capitalize()}: {status}")
        
        # Log recommended models
        embedding_model = self.get_recommended_embedding_model()
        generation_model = self.get_recommended_generation_model()
        
        logger.info(f"  Recommended embedding model: {embedding_model}")
        logger.info(f"  Recommended generation model: {generation_model}")


def get_model_manager() -> ModelManager:
    """Get a singleton instance of the model manager."""
    if not hasattr(get_model_manager, "_instance"):
        get_model_manager._instance = ModelManager()
    return get_model_manager._instance
