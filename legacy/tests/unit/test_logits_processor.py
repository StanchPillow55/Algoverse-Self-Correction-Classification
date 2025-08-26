"""
Unit tests for the LogitsProcessor class.

Tests logits processing and feature extraction functionality.
"""

import pytest
import sys
import numpy as np
from pathlib import Path

# Add src to path for testing
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.embeddings.logits_processor import LogitsProcessor
except ImportError:
    pytest.skip("LogitsProcessor not available", allow_module_level=True)


class TestLogitsProcessor:
    """Test the LogitsProcessor functionality."""

    def test_logits_processor_initialization(self):
        """Test that LogitsProcessor can be initialized."""
        processor = LogitsProcessor(top_k=5, include_entropy=True)
        assert processor.top_k == 5
        assert processor.include_entropy is True

    def test_process_single_logits(self):
        """Test processing of single logits array."""
        processor = LogitsProcessor(top_k=3, include_entropy=True)
        
        # Create mock logits
        logits = np.array([1.0, 2.0, 0.5, 3.0, 1.5])
        
        features = processor.process_logits(logits)
        
        # Check that features are generated
        assert isinstance(features, dict)
        assert 'max_logit' in features
        assert 'entropy' in features
        assert 'top_1_prob' in features
        assert features['max_logit'] == 3.0

    def test_process_sequence_logits(self):
        """Test processing of sequence logits."""
        processor = LogitsProcessor(top_k=3, include_entropy=True)
        
        # Create mock sequence logits
        sequence_logits = [
            np.array([1.0, 2.0, 0.5]),
            np.array([0.8, 1.5, 2.2]),
            np.array([2.1, 1.0, 0.3])
        ]
        
        features = processor.process_sequence_logits(sequence_logits)
        
        # Check that sequence features are generated
        assert isinstance(features, dict)
        assert 'sequence_length' in features
        assert 'avg_uncertainty' in features
        assert features['sequence_length'] == 3

    def test_create_logits_features_matrix(self):
        """Test creation of features matrix from multiple samples."""
        processor = LogitsProcessor(top_k=3, include_entropy=True)
        
        # Create mock logits data for multiple samples
        logits_data = [
            np.array([1.0, 2.0, 0.5, 3.0]),
            np.array([0.8, 1.5, 2.2, 1.1]),
            np.array([2.1, 1.0, 0.3, 2.5])
        ]
        
        feature_matrix = processor.create_logits_features_matrix(logits_data)
        
        # Check matrix properties
        assert isinstance(feature_matrix, np.ndarray)
        assert feature_matrix.shape[0] == 3  # 3 samples
        assert feature_matrix.shape[1] > 0   # Some features

    def test_get_feature_names(self):
        """Test getting feature names."""
        processor = LogitsProcessor(top_k=3, include_entropy=True)
        
        feature_names = processor.get_feature_names()
        
        assert isinstance(feature_names, list)
        assert len(feature_names) > 0
        assert 'top_1_prob' in feature_names
        assert 'entropy' in feature_names

    def test_empty_logits_handling(self):
        """Test handling of empty logits."""
        processor = LogitsProcessor(top_k=3, include_entropy=True)
        
        # Test empty array
        empty_logits = np.array([])
        features = processor.process_logits(empty_logits)
        
        # Should return default features
        assert isinstance(features, dict)
        assert all(value == 0.0 for value in features.values())

    def test_softmax_calculation(self):
        """Test internal softmax calculation."""
        processor = LogitsProcessor()
        
        logits = np.array([1.0, 2.0, 3.0])
        probabilities = processor._softmax(logits)
        
        # Check that probabilities sum to 1
        assert np.isclose(np.sum(probabilities), 1.0)
        assert all(prob >= 0 for prob in probabilities)

    def test_entropy_calculation(self):
        """Test entropy calculation."""
        processor = LogitsProcessor()
        
        # Uniform distribution should have high entropy
        uniform_probs = np.array([0.25, 0.25, 0.25, 0.25])
        uniform_entropy = processor._calculate_entropy(uniform_probs)
        
        # Concentrated distribution should have low entropy
        concentrated_probs = np.array([0.9, 0.05, 0.03, 0.02])
        concentrated_entropy = processor._calculate_entropy(concentrated_probs)
        
        assert uniform_entropy > concentrated_entropy


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
