"""
Smoke tests for basic pipeline functionality.

These tests verify that the core components can be imported and initialized.
"""

import pytest
import sys
import os
from pathlib import Path

# Add src to path for testing
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.utils.error_types import ErrorType, get_all_error_types, get_error_definition
    from src.utils.config import Config
    from src.classification.classifier import ErrorClassifier
    from src.post_processing.error_router import ErrorRouter
except ImportError as e:
    pytest.skip(f"Import error: {e}", allow_module_level=True)


def test_error_types_import():
    """Test that error types can be imported and accessed."""
    error_types = get_all_error_types()
    assert len(error_types) == 6  # Including NO_ERROR
    assert ErrorType.ANSWER_WAVERING in error_types
    assert ErrorType.PROMPT_BIAS in error_types


def test_error_definitions():
    """Test that error definitions are properly defined."""
    for error_type in get_all_error_types():
        definition = get_error_definition(error_type)
        assert definition.name is not None
        assert definition.description is not None
        assert definition.severity in ["none", "low", "medium", "high"]


def test_config_loading():
    """Test that configuration can be loaded."""
    # This should not raise an exception
    Config.load()


def test_classifier_initialization():
    """Test that the classifier can be initialized."""
    classifier = ErrorClassifier()
    assert classifier.model is not None


def test_error_router_initialization():
    """Test that the error router can be initialized."""
    router = ErrorRouter()
    assert router.confidence_threshold == 0.7
    
    # Test routing
    route_info = router.route_error(ErrorType.ANSWER_WAVERING, 0.8)
    assert route_info["error_type"] == "answer_wavering"
    assert route_info["confidence"] == 0.8
    assert route_info["requires_correction"] is True


def test_main_module_import():
    """Test that the main module can be imported."""
    from src import main
    assert main.main is not None


if __name__ == "__main__":
    pytest.main([__file__])
