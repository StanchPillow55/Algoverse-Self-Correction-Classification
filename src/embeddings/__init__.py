"""
Embedding generation module for semantic analysis of LLM outputs.

This module handles:
- Text embedding generation using various models
- Semantic similarity computation
- Feature extraction for classification
"""

from .embedding_generator import EmbeddingGenerator
from .semantic_analyzer import SemanticAnalyzer

__all__ = ["EmbeddingGenerator", "SemanticAnalyzer"]
