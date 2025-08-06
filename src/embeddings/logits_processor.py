"""
Logits processor for handling Llama model logits as features.

Processes and extracts meaningful features from model logits for 
integration into the classification pipeline.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)


class LogitsProcessor:
    """Process and extract features from Llama model logits."""
    
    def __init__(self, top_k: int = 10, include_entropy: bool = True):
        """
        Initialize the logits processor.
        
        Args:
            top_k: Number of top tokens to consider for feature extraction
            include_entropy: Whether to include entropy-based features
        """
        self.top_k = top_k
        self.include_entropy = include_entropy
        logger.info(f"Initialized LogitsProcessor with top_k={top_k}")
    
    def process_logits(self, logits: Union[List[float], np.ndarray]) -> Dict[str, float]:
        """
        Process raw logits into meaningful features.
        
        Args:
            logits: Raw logits array from Llama model
            
        Returns:
            Dictionary of extracted features
        """
        if isinstance(logits, list):
            logits = np.array(logits)
            
        if logits.size == 0:
            logger.warning("Empty logits array provided")
            return self._get_empty_features()
        
        # Convert to probabilities
        probabilities = self._softmax(logits)
        
        features = {}
        
        # Top-k probabilities
        top_k_indices = np.argsort(logits)[-self.top_k:]
        top_k_probs = probabilities[top_k_indices]
        
        for i, prob in enumerate(top_k_probs):
            features[f'top_{i+1}_prob'] = float(prob)
        
        # Statistical features
        features.update({
            'max_logit': float(np.max(logits)),
            'min_logit': float(np.min(logits)),
            'mean_logit': float(np.mean(logits)),
            'std_logit': float(np.std(logits)),
            'max_prob': float(np.max(probabilities)),
            'prob_mass_top_k': float(np.sum(top_k_probs))
        })
        
        # Entropy-based features
        if self.include_entropy:
            entropy = self._calculate_entropy(probabilities)
            features.update({
                'entropy': float(entropy),
                'normalized_entropy': float(entropy / np.log(len(probabilities))),
                'confidence': float(1.0 - (entropy / np.log(len(probabilities))))
            })
        
        # Concentration measures
        features.update({
            'gini_coefficient': float(self._calculate_gini(probabilities)),
            'effective_vocab_size': float(self._calculate_effective_vocab_size(probabilities))
        })
        
        return features
    
    def process_sequence_logits(self, sequence_logits: List[Union[List[float], np.ndarray]]) -> Dict[str, float]:
        """
        Process logits from a sequence of tokens.
        
        Args:
            sequence_logits: List of logits arrays for each token in sequence
            
        Returns:
            Dictionary of aggregated sequence-level features
        """
        if not sequence_logits:
            return self._get_empty_features()
        
        # Process individual token logits
        token_features = []
        for token_logits in sequence_logits:
            token_features.append(self.process_logits(token_logits))
        
        # Aggregate features across sequence
        aggregated_features = {}
        
        if token_features:
            # Get feature names from first token
            feature_names = token_features[0].keys()
            
            for feature_name in feature_names:
                values = [tf[feature_name] for tf in token_features]
                
                # Aggregate statistics
                aggregated_features.update({
                    f'seq_mean_{feature_name}': float(np.mean(values)),
                    f'seq_std_{feature_name}': float(np.std(values)),
                    f'seq_min_{feature_name}': float(np.min(values)),
                    f'seq_max_{feature_name}': float(np.max(values))
                })
            
            # Sequence-specific features
            aggregated_features.update({
                'sequence_length': len(sequence_logits),
                'avg_uncertainty': float(np.mean([tf.get('entropy', 0) for tf in token_features])),
                'uncertainty_variance': float(np.var([tf.get('entropy', 0) for tf in token_features]))
            })
        
        return aggregated_features
    
    def create_logits_features_matrix(self, logits_data: List[Union[Dict, List, np.ndarray]]) -> np.ndarray:
        """
        Create a feature matrix from logits data for multiple samples.
        
        Args:
            logits_data: List of logits data (can be raw logits, processed features, or sequences)
            
        Returns:
            Feature matrix (n_samples, n_features)
        """
        features_list = []
        
        for sample_logits in logits_data:
            if isinstance(sample_logits, dict):
                # Already processed features
                features = sample_logits
            elif isinstance(sample_logits, (list, np.ndarray)):
                if len(sample_logits) > 0 and isinstance(sample_logits[0], (list, np.ndarray)):
                    # Sequence of logits
                    features = self.process_sequence_logits(sample_logits)
                else:
                    # Single logits array
                    features = self.process_logits(sample_logits)
            else:
                logger.warning(f"Unknown logits data type: {type(sample_logits)}")
                features = self._get_empty_features()
            
            features_list.append(features)
        
        # Convert to matrix
        if not features_list:
            return np.array([])
        
        # Get all unique feature names
        all_features = set()
        for features in features_list:
            all_features.update(features.keys())
        
        feature_names = sorted(list(all_features))
        
        # Create matrix
        matrix = np.zeros((len(features_list), len(feature_names)))
        for i, features in enumerate(features_list):
            for j, feature_name in enumerate(feature_names):
                matrix[i, j] = features.get(feature_name, 0.0)
        
        logger.info(f"Created logits feature matrix with shape {matrix.shape}")
        return matrix
    
    def get_feature_names(self) -> List[str]:
        """Get the names of features that will be extracted."""
        feature_names = []
        
        # Top-k probability features
        for i in range(1, self.top_k + 1):
            feature_names.append(f'top_{i}_prob')
        
        # Statistical features
        feature_names.extend([
            'max_logit', 'min_logit', 'mean_logit', 'std_logit',
            'max_prob', 'prob_mass_top_k'
        ])
        
        # Entropy features
        if self.include_entropy:
            feature_names.extend(['entropy', 'normalized_entropy', 'confidence'])
        
        # Concentration features
        feature_names.extend(['gini_coefficient', 'effective_vocab_size'])
        
        return feature_names
    
    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        """Apply softmax to logits."""
        exp_logits = np.exp(logits - np.max(logits))  # Subtract max for numerical stability
        return exp_logits / np.sum(exp_logits)
    
    def _calculate_entropy(self, probabilities: np.ndarray) -> float:
        """Calculate entropy of probability distribution."""
        # Add small epsilon to avoid log(0)
        epsilon = 1e-12
        probs = np.clip(probabilities, epsilon, 1.0)
        return -np.sum(probs * np.log(probs))
    
    def _calculate_gini(self, probabilities: np.ndarray) -> float:
        """Calculate Gini coefficient of probability distribution."""
        sorted_probs = np.sort(probabilities)
        n = len(sorted_probs)
        index = np.arange(1, n + 1)
        return (2 * np.sum(index * sorted_probs)) / (n * np.sum(sorted_probs)) - (n + 1) / n
    
    def _calculate_effective_vocab_size(self, probabilities: np.ndarray) -> float:
        """Calculate effective vocabulary size (perplexity)."""
        entropy = self._calculate_entropy(probabilities)
        return np.exp(entropy)
    
    def _get_empty_features(self) -> Dict[str, float]:
        """Get a dictionary of empty/default features."""
        features = {}
        
        # Top-k features
        for i in range(1, self.top_k + 1):
            features[f'top_{i}_prob'] = 0.0
        
        # Statistical features
        features.update({
            'max_logit': 0.0,
            'min_logit': 0.0,
            'mean_logit': 0.0,
            'std_logit': 0.0,
            'max_prob': 0.0,
            'prob_mass_top_k': 0.0
        })
        
        # Entropy features
        if self.include_entropy:
            features.update({
                'entropy': 0.0,
                'normalized_entropy': 0.0,
                'confidence': 0.0
            })
        
        # Concentration features
        features.update({
            'gini_coefficient': 0.0,
            'effective_vocab_size': 0.0
        })
        
        return features
