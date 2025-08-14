"""
Classification module for LLM error type prediction.

This module handles:
- Multi-class error classification
- Model training and evaluation
- Prediction confidence scoring
"""

from .classifier import ErrorClassifier
from .model_trainer import ModelTrainer
from .evaluator import ClassificationEvaluator

__all__ = ["ErrorClassifier", "ModelTrainer", "ClassificationEvaluator"]
