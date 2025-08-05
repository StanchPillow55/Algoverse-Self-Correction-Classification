"""
Post-processing module for routing errors to correction strategies.

This module handles:
- Error prediction routing
- Re-prompting strategy selection
- Correction application
"""

from .error_router import ErrorRouter
from .correction_strategies import CorrectionStrategies

__all__ = ["ErrorRouter", "CorrectionStrategies"]
