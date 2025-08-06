"""
Feature fusion module for combining embeddings with logits features.

Provides flexible methods to combine semantic embeddings with logits-derived
features for enhanced classification performance.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class FusionMethod(Enum):
    """Available methods for feature fusion."""
    CONCATENATION = "concatenation"
    WEIGHTED_CONCATENATION = "weighted_concatenation"
    ELEMENT_WISE = "element_wise"
    ATTENTION_FUSION = "attention_fusion"


class FeatureFusion:
    """Handles fusion of embeddings and logits features."""
    
    def __init__(self, fusion_method: Union[str, FusionMethod] = FusionMethod.CONCATENATION):
        """
        Initialize the feature fusion module.
        
        Args:
            fusion_method: Method to use for combining features
        """
        if isinstance(fusion_method, str):
            fusion_method = FusionMethod(fusion_method)
        
        self.fusion_method = fusion_method
        self.embedding_weight = 1.0
        self.logits_weight = 1.0
        
        logger.info(f"Initialized FeatureFusion with method: {fusion_method.value}")
    
    def fuse_features(self, 
                     embeddings: Optional[np.ndarray] = None,
                     logits_features: Optional[np.ndarray] = None,
                     text_features: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Fuse different types of features into a single feature matrix.
        
        Args:
            embeddings: Semantic embeddings matrix (n_samples, embedding_dim)
            logits_features: Logits-derived features (n_samples, logits_features_dim)
            text_features: Text statistics features (n_samples, text_features_dim)
            
        Returns:
            Fused feature matrix (n_samples, fused_dim)
        """
        available_features = []
        feature_names = []
        
        # Collect available features
        if embeddings is not None and embeddings.size > 0:
            available_features.append(("embeddings", embeddings))
            feature_names.extend([f"embed_{i}" for i in range(embeddings.shape[1])])
        
        if logits_features is not None and logits_features.size > 0:
            available_features.append(("logits", logits_features))
            feature_names.extend([f"logits_{i}" for i in range(logits_features.shape[1])])
        
        if text_features is not None and text_features.size > 0:
            available_features.append(("text", text_features))
            feature_names.extend([f"text_{i}" for i in range(text_features.shape[1])])
        
        if not available_features:
            logger.warning("No features provided for fusion")
            return np.array([])
        
        if len(available_features) == 1:
            logger.info(f"Only one feature type available: {available_features[0][0]}")
            return available_features[0][1]
        
        # Apply fusion method
        if self.fusion_method == FusionMethod.CONCATENATION:
            return self._concatenate_features(available_features)
        elif self.fusion_method == FusionMethod.WEIGHTED_CONCATENATION:
            return self._weighted_concatenate_features(available_features)
        elif self.fusion_method == FusionMethod.ELEMENT_WISE:
            return self._element_wise_fusion(available_features)
        elif self.fusion_method == FusionMethod.ATTENTION_FUSION:
            return self._attention_fusion(available_features)
        else:
            logger.warning(f"Unknown fusion method: {self.fusion_method}")
            return self._concatenate_features(available_features)
    
    def set_weights(self, embedding_weight: float = 1.0, logits_weight: float = 1.0):
        """
        Set weights for weighted fusion methods.
        
        Args:
            embedding_weight: Weight for embedding features
            logits_weight: Weight for logits features
        """
        self.embedding_weight = embedding_weight
        self.logits_weight = logits_weight
        logger.info(f"Updated fusion weights: embeddings={embedding_weight}, logits={logits_weight}")
    
    def get_fused_feature_names(self, 
                               embedding_dim: int = 0,
                               logits_feature_names: Optional[List[str]] = None,
                               text_feature_names: Optional[List[str]] = None) -> List[str]:
        """
        Get names for fused features.
        
        Args:
            embedding_dim: Dimension of embedding features
            logits_feature_names: Names of logits features
            text_feature_names: Names of text features
            
        Returns:
            List of fused feature names
        """
        feature_names = []
        
        # Embedding features
        if embedding_dim > 0:
            feature_names.extend([f"embed_{i}" for i in range(embedding_dim)])
        
        # Logits features
        if logits_feature_names:
            feature_names.extend([f"logits_{name}" for name in logits_feature_names])
        
        # Text features
        if text_feature_names:
            feature_names.extend([f"text_{name}" for name in text_feature_names])
        
        return feature_names
    
    def _concatenate_features(self, feature_list: List[Tuple[str, np.ndarray]]) -> np.ndarray:
        """Simple concatenation of all features."""
        matrices = [features for _, features in feature_list]
        fused = np.concatenate(matrices, axis=1)
        
        logger.info(f"Concatenated features: {[f.shape for _, f in feature_list]} -> {fused.shape}")
        return fused
    
    def _weighted_concatenate_features(self, feature_list: List[Tuple[str, np.ndarray]]) -> np.ndarray:
        """Weighted concatenation of features."""
        weighted_matrices = []
        
        for feature_type, features in feature_list:
            if feature_type == "embeddings":
                weighted_features = features * self.embedding_weight
            elif feature_type == "logits":
                weighted_features = features * self.logits_weight
            else:
                weighted_features = features  # No weighting for other types
            
            weighted_matrices.append(weighted_features)
        
        fused = np.concatenate(weighted_matrices, axis=1)
        
        logger.info(f"Weighted concatenated features: {[f.shape for _, f in feature_list]} -> {fused.shape}")
        return fused
    
    def _element_wise_fusion(self, feature_list: List[Tuple[str, np.ndarray]]) -> np.ndarray:
        """
        Element-wise fusion (requires same dimensions or broadcasting).
        Only works if features have compatible shapes.
        """
        if len(feature_list) < 2:
            return feature_list[0][1]
        
        # Start with first feature matrix
        fused = feature_list[0][1].copy()
        
        for i in range(1, len(feature_list)):
            _, features = feature_list[i]
            
            # Try element-wise combination
            try:
                # Simple average for now - could use more sophisticated methods
                if fused.shape == features.shape:
                    fused = (fused + features) / 2.0
                else:
                    logger.warning(f"Shape mismatch for element-wise fusion: {fused.shape} vs {features.shape}")
                    # Fall back to concatenation
                    fused = np.concatenate([fused, features], axis=1)
            except Exception as e:
                logger.warning(f"Element-wise fusion failed: {e}. Falling back to concatenation.")
                fused = np.concatenate([fused, features], axis=1)
        
        logger.info(f"Element-wise fused features shape: {fused.shape}")
        return fused
    
    def _attention_fusion(self, feature_list: List[Tuple[str, np.ndarray]]) -> np.ndarray:
        """
        Simple attention-based fusion.
        Computes attention weights based on feature variance.
        """
        if len(feature_list) < 2:
            return feature_list[0][1]
        
        # Calculate attention weights based on feature variance
        attention_weights = []
        for _, features in feature_list:
            # Higher variance features get higher attention
            variance = np.var(features, axis=0).mean()
            attention_weights.append(variance)
        
        # Normalize weights
        total_weight = sum(attention_weights)
        if total_weight > 0:
            attention_weights = [w / total_weight for w in attention_weights]
        else:
            attention_weights = [1.0 / len(feature_list)] * len(feature_list)
        
        # Apply attention weights and concatenate
        weighted_features = []
        for (_, features), weight in zip(feature_list, attention_weights):
            weighted_features.append(features * weight)
        
        fused = np.concatenate(weighted_features, axis=1)
        
        logger.info(f"Attention-fused features with weights {attention_weights}: {fused.shape}")
        return fused
    
    def validate_feature_compatibility(self, 
                                     embeddings: Optional[np.ndarray] = None,
                                     logits_features: Optional[np.ndarray] = None,
                                     text_features: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Validate that features are compatible for fusion.
        
        Returns:
            Dictionary with validation results
        """
        validation_result = {
            "is_valid": True,
            "warnings": [],
            "feature_info": {}
        }
        
        feature_matrices = []
        if embeddings is not None:
            feature_matrices.append(("embeddings", embeddings))
        if logits_features is not None:
            feature_matrices.append(("logits", logits_features))
        if text_features is not None:
            feature_matrices.append(("text", text_features))
        
        if not feature_matrices:
            validation_result["is_valid"] = False
            validation_result["warnings"].append("No features provided")
            return validation_result
        
        # Check sample count consistency
        n_samples_list = [features.shape[0] for _, features in feature_matrices]
        if len(set(n_samples_list)) > 1:
            validation_result["is_valid"] = False
            validation_result["warnings"].append(f"Inconsistent sample counts: {n_samples_list}")
        
        # Record feature information
        for name, features in feature_matrices:
            validation_result["feature_info"][name] = {
                "shape": features.shape,
                "dtype": str(features.dtype),
                "has_nan": bool(np.isnan(features).any()),
                "has_inf": bool(np.isinf(features).any())
            }
            
            # Check for problematic values
            if np.isnan(features).any():
                validation_result["warnings"].append(f"{name} contains NaN values")
            if np.isinf(features).any():
                validation_result["warnings"].append(f"{name} contains infinite values")
        
        return validation_result
