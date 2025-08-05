"""
Error type definitions for LLM output classification.

Based on "Understanding the Dark Side of LLMs Intrinsic Self-Correction" research.
"""

from enum import Enum
from typing import Dict, List
from dataclasses import dataclass


class ErrorType(Enum):
    """Enumeration of LLM error types."""
    
    ANSWER_WAVERING = "answer_wavering"
    PROMPT_BIAS = "prompt_bias"
    OVERTHINKING = "overthinking"
    COGNITIVE_OVERLOAD = "cognitive_overload"
    PERFECTIONISM_BIAS = "perfectionism_bias"
    NO_ERROR = "no_error"


@dataclass
class ErrorDefinition:
    """Definition and characteristics of an error type."""
    
    name: str
    description: str
    symptoms: List[str]
    severity: str  # "low", "medium", "high"
    correction_strategy: str


# Error type definitions based on research literature
ERROR_DEFINITIONS: Dict[ErrorType, ErrorDefinition] = {
    ErrorType.ANSWER_WAVERING: ErrorDefinition(
        name="Answer Wavering",
        description="Model changes its answer multiple times without clear justification",
        symptoms=[
            "Multiple contradicting statements",
            "Frequent use of hedging language",
            "Back-and-forth reasoning",
            "Uncertainty markers without basis"
        ],
        severity="medium",
        correction_strategy="confidence_anchoring"
    ),
    
    ErrorType.PROMPT_BIAS: ErrorDefinition(
        name="Prompt Bias",
        description="Model responses are heavily influenced by prompt framing rather than content",
        symptoms=[
            "Echoing prompt language unnecessarily",
            "Assumptions based on prompt tone",
            "Ignoring conflicting evidence",
            "Over-alignment with prompt expectations"
        ],
        severity="high",
        correction_strategy="neutral_reframing"
    ),
    
    ErrorType.OVERTHINKING: ErrorDefinition(
        name="Overthinking",
        description="Model provides unnecessarily complex solutions to simple problems",
        symptoms=[
            "Excessive elaboration",
            "Introducing irrelevant complexity",
            "Over-detailed explanations",
            "Analysis paralysis indicators"
        ],
        severity="low",
        correction_strategy="simplification_prompt"
    ),
    
    ErrorType.COGNITIVE_OVERLOAD: ErrorDefinition(
        name="Cognitive Overload",
        description="Model struggles with complex multi-step reasoning",
        symptoms=[
            "Losing track of context",
            "Incomplete reasoning chains",
            "Contradictory intermediate steps",
            "Failure to maintain coherence"
        ],
        severity="high",
        correction_strategy="step_by_step_decomposition"
    ),
    
    ErrorType.PERFECTIONISM_BIAS: ErrorDefinition(
        name="Perfectionism Bias",
        description="Model over-corrects or provides unnecessarily perfect solutions",
        symptoms=[
            "Excessive qualification",
            "Over-cautious language",
            "Unnecessary disclaimers",
            "Avoidance of definitive statements"
        ],
        severity="medium",
        correction_strategy="confidence_calibration"
    ),
    
    ErrorType.NO_ERROR: ErrorDefinition(
        name="No Error",
        description="Model output appears correct and well-reasoned",
        symptoms=[],
        severity="none",
        correction_strategy="none"
    )
}


def get_error_definition(error_type: ErrorType) -> ErrorDefinition:
    """Get the definition for a specific error type."""
    return ERROR_DEFINITIONS[error_type]


def get_all_error_types() -> List[ErrorType]:
    """Get all available error types."""
    return list(ErrorType)


def get_correction_strategies() -> Dict[str, str]:
    """Get mapping of correction strategies to their descriptions."""
    return {
        "confidence_anchoring": "Provide clear confidence levels and reasoning",
        "neutral_reframing": "Rephrase prompts to reduce bias",
        "simplification_prompt": "Request simpler, more direct answers",
        "step_by_step_decomposition": "Break complex problems into smaller steps",
        "confidence_calibration": "Balance confidence with appropriate uncertainty",
        "none": "No correction needed"
    }
