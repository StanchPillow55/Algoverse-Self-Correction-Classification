"""
Unit tests for the ModelManager class.

Tests model availability detection and recommendations.
"""

import pytest
import sys
from pathlib import Path

# Add src to path for testing
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.utils.model_manager import ModelManager, get_model_manager
except ImportError:
    pytest.skip("ModelManager not available", allow_module_level=True)


class TestModelManager:
    """Test the ModelManager functionality."""

    def test_model_manager_initialization(self):
        """Test that ModelManager can be initialized."""
        manager = ModelManager()
        assert manager is not None
        
        # Check that models are defined
        models = manager.get_available_models()
        assert "local" in models
        assert "api" in models
        assert "embedding" in models["local"]

    def test_get_recommended_models(self):
        """Test that recommended models are returned."""
        manager = ModelManager()
        
        embedding_model = manager.get_recommended_embedding_model()
        generation_model = manager.get_recommended_generation_model()
        
        assert embedding_model is not None
        assert generation_model is not None

    def test_model_config(self):
        """Test model configuration retrieval."""
        manager = ModelManager()
        
        # Test local model config
        local_model = "sentence-transformers/all-MiniLM-L6-v2"
        config = manager.get_model_config(local_model)
        
        assert config["name"] == local_model
        assert config["type"] == "local"
        assert config["available"] is True

    def test_singleton_model_manager(self):
        """Test that get_model_manager returns a singleton."""
        manager1 = get_model_manager()
        manager2 = get_model_manager()
        
        assert manager1 is manager2

    def test_api_availability_detection(self):
        """Test API availability detection (should be False without keys)."""
        manager = ModelManager()
        
        # Without API keys, these should return False
        assert manager.is_api_available("openai") is False
        assert manager.is_api_available("anthropic") is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
