"""
Unit tests for the FeatureFusion class.

Tests feature fusion methods and validation functionality.
"""

import pytest
import sys
import numpy as np
from pathlib import Path

# Add src to path for testing
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.embeddings.feature_fusion import FeatureFusion, FusionMethod
except ImportError:
    pytest.skip("FeatureFusion not available", allow_module_level=True)


class TestFeatureFusion:
    """Test the FeatureFusion functionality."""

    def test_feature_fusion_initialization(self):
        """Test that FeatureFusion can be initialized."""
        fusion = FeatureFusion(fusion_method="concatenation")
        assert fusion.fusion_method == FusionMethod.CONCATENATION

    def test_concatenation_fusion(self):
        """Test simple concatenation of features."""
        fusion = FeatureFusion(fusion_method="concatenation")
        
        # Create mock features
        embeddings = np.random.rand(10, 5)  # 10 samples, 5 embedding features
        text_features = np.random.rand(10, 3)  # 10 samples, 3 text features
        
        fused = fusion.fuse_features(
            embeddings=embeddings,
            text_features=text_features
        )
        
        # Check that features were concatenated
        assert fused.shape == (10, 8)  # 5 + 3 features

    def test_weighted_concatenation_fusion(self):
        """Test weighted concatenation of features."""
        fusion = FeatureFusion(fusion_method="weighted_concatenation")
        fusion.set_weights(embedding_weight=2.0, logits_weight=0.5)
        
        embeddings = np.ones((5, 3))
        logits_features = np.ones((5, 2))
        
        fused = fusion.fuse_features(
            embeddings=embeddings,
            logits_features=logits_features
        )
        
        # Check that features were weighted and concatenated
        assert fused.shape == (5, 5)  # 3 + 2 features
        
        # Check that embedding features are weighted by 2.0
        assert np.allclose(fused[:, :3], 2.0)
        # Check that logits features are weighted by 0.5
        assert np.allclose(fused[:, 3:], 0.5)

    def test_single_feature_type(self):
        """Test fusion with only one feature type."""
        fusion = FeatureFusion()
        
        embeddings = np.random.rand(5, 4)
        
        fused = fusion.fuse_features(embeddings=embeddings)
        
        # Should return the same embeddings
        assert np.array_equal(fused, embeddings)

    def test_no_features_provided(self):
        """Test handling when no features are provided."""
        fusion = FeatureFusion()
        
        fused = fusion.fuse_features()
        
        # Should return empty array
        assert fused.size == 0

    def test_feature_validation(self):
        """Test feature compatibility validation."""
        fusion = FeatureFusion()
        
        # Compatible features (same number of samples)
        embeddings = np.random.rand(10, 5)
        text_features = np.random.rand(10, 3)
        
        validation = fusion.validate_feature_compatibility(
            embeddings=embeddings,
            text_features=text_features
        )
        
        assert validation["is_valid"] is True
        assert "embeddings" in validation["feature_info"]
        assert "text" in validation["feature_info"]

    def test_incompatible_sample_counts(self):
        """Test validation with incompatible sample counts."""
        fusion = FeatureFusion()
        
        # Incompatible features (different number of samples)
        embeddings = np.random.rand(10, 5)
        text_features = np.random.rand(8, 3)
        
        validation = fusion.validate_feature_compatibility(
            embeddings=embeddings,
            text_features=text_features
        )
        
        assert validation["is_valid"] is False
        assert len(validation["warnings"]) > 0

    def test_get_fused_feature_names(self):
        """Test getting fused feature names."""
        fusion = FeatureFusion()
        
        feature_names = fusion.get_fused_feature_names(
            embedding_dim=3,
            logits_feature_names=["entropy", "max_prob"],
            text_feature_names=["length", "word_count"]
        )
        
        expected_names = [
            "embed_0", "embed_1", "embed_2",
            "logits_entropy", "logits_max_prob",
            "text_length", "text_word_count"
        ]
        
        assert feature_names == expected_names

    def test_attention_fusion(self):
        """Test attention-based fusion method."""
        fusion = FeatureFusion(fusion_method="attention_fusion")
        
        # Create features with different variances
        low_var_features = np.ones((5, 2)) * 0.1  # Low variance
        high_var_features = np.random.rand(5, 2) * 10  # High variance
        
        fused = fusion.fuse_features(
            embeddings=low_var_features,
            logits_features=high_var_features
        )
        
        # Should have concatenated features
        assert fused.shape == (5, 4)

    def test_element_wise_fusion_same_shape(self):
        """Test element-wise fusion with same shaped features."""
        fusion = FeatureFusion(fusion_method="element_wise")
        
        features1 = np.ones((3, 4))
        features2 = np.ones((3, 4)) * 2
        
        fused = fusion.fuse_features(
            embeddings=features1,
            logits_features=features2
        )
        
        # Should average the features
        assert fused.shape == (3, 4)
        assert np.allclose(fused, 1.5)  # Average of 1 and 2

    def test_element_wise_fusion_different_shapes(self):
        """Test element-wise fusion falls back to concatenation for different shapes."""
        fusion = FeatureFusion(fusion_method="element_wise")
        
        features1 = np.ones((3, 2))
        features2 = np.ones((3, 4))
        
        fused = fusion.fuse_features(
            embeddings=features1,
            logits_features=features2
        )
        
        # Should fall back to concatenation
        assert fused.shape == (3, 6)  # 2 + 4

    def test_nan_and_inf_detection(self):
        """Test detection of NaN and infinite values."""
        fusion = FeatureFusion()
        
        # Create features with problematic values
        bad_features = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, np.inf]])
        good_features = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        
        validation = fusion.validate_feature_compatibility(
            embeddings=bad_features,
            text_features=good_features
        )
        
        # Should detect NaN and inf values
        assert "embeddings contains NaN values" in validation["warnings"]
        assert "embeddings contains infinite values" in validation["warnings"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
