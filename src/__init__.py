"""
LLM Error Classification Pipeline

A comprehensive pipeline for identifying and correcting errors in Large Language Model outputs.
Inspired by "Understanding the Dark Side of LLMs Intrinsic Self-Correction" (Sharma et al., 2023).

Error Types Supported:
- Answer Wavering
- Prompt Bias
- Overthinking
- Cognitive Overload
- Perfectionism Bias
"""

__version__ = "0.1.0"
__author__ = "Algoverse Research Team"

from .main import main
from .utils.error_types import ErrorType
from .utils.config import Config

__all__ = ["main", "ErrorType", "Config"]
