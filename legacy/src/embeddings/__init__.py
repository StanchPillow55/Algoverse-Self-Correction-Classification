"""
Embedding generation module for semantic analysis of LLM outputs.

This module handles:
- Text embedding generation using various models
- Semantic similarity computation
- Feature extraction for classification
"""

from .embedding_generator import EmbeddingGenerator
from .semantic_analyzer import SemanticAnalyzer
from .logits_processor import LogitsProcessor
from .feature_fusion import FeatureFusion, FusionMethod

__all__ = ["EmbeddingGenerator", "SemanticAnalyzer", "LogitsProcessor", "FeatureFusion", "FusionMethod"]
