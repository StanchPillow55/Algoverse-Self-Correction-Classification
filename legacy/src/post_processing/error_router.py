"""
Error router for directing predictions to appropriate correction strategies.

Routes classified errors to their corresponding re-prompting strategies.
"""

from typing import Dict, Any
from ..utils.error_types import ErrorType, get_error_definition
import logging

logger = logging.getLogger(__name__)


class ErrorRouter:
    """Routes error predictions to appropriate correction strategies."""
    
    def __init__(self, confidence_threshold: float = 0.7):
        """Initialize the error router with a confidence threshold."""
        self.confidence_threshold = confidence_threshold
        
    def route_error(self, error_type: ErrorType, confidence: float) -> Dict[str, Any]:
        """Route an error prediction to the appropriate correction strategy."""
        error_def = get_error_definition(error_type)
        
        route_info = {
            "error_type": error_type.value,
            "confidence": confidence,
            "correction_strategy": error_def.correction_strategy,
            "severity": error_def.severity,
            "requires_correction": confidence >= self.confidence_threshold
        }
        
        if route_info["requires_correction"]:
            logger.info(f"Routing {error_type.value} to {error_def.correction_strategy}")
        else:
            logger.info(f"Low confidence ({confidence}) for {error_type.value}, no correction applied")
            
        return route_info
    
    def set_confidence_threshold(self, threshold: float):
        """Update the confidence threshold for corrections."""
        self.confidence_threshold = threshold
        logger.info(f"Updated confidence threshold to {threshold}")
