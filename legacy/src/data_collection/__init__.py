"""
Data collection module for LLM error classification.

This module handles:
- Synthetic data generation using LLMs (GPT-4, Claude, Llama)
- Annotation of error types
- Dataset curation and validation
"""

from .synthetic_generator import SyntheticDataGenerator
from .annotator import ErrorAnnotator
from .dataset_curator import DatasetCurator
from .preprocessor import DataPreprocessor

__all__ = ["SyntheticDataGenerator", "ErrorAnnotator", "DatasetCurator", "DataPreprocessor"]
