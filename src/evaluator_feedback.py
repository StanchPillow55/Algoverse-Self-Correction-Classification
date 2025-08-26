"""
Evaluator feedback system for converting bias labels to actionable coaching.

Provides coaching feedback that helps learners understand and correct 
their cognitive biases in problem-solving.
"""

from typing import Optional, Dict

# Mapping from bias labels to coaching feedback (max 2 sentences, second person)
BIAS_COACHING: Dict[str, str] = {
    "Confirmation": (
        "You are hyper-confirming your training data instead of solving the problem. "
        "Pause and derive the answer from first principles, then recompute the key step."
    ),
    "Anchoring": (
        "You are anchoring on numbers or phrases from the problem statement. "
        "Ignore the surface features and work through the logic step by step."
    ),
    "Fixation": (
        "You are fixated on your initial approach and missing simpler solutions. "
        "Step back, consider alternative methods, and question your assumptions."
    ),
    "Overconfidence": (
        "You are overconfident in your answer without proper verification. "
        "Double-check your work and consider where you might have made errors."
    ),
    "SunkCost": (
        "You are persisting with a flawed approach because you've invested effort. "
        "Cut your losses and try a completely different strategy."
    ),
    "Availability": (
        "You are defaulting to recent examples instead of the current problem. "
        "Focus on the specific details and requirements of this particular question."
    ),
    "Availability/Bandwagon": (
        "You are following common patterns instead of analyzing this specific case. "
        "Think independently about what this problem actually requires."
    ),
    "Outcome": (
        "You are working backwards from an assumed outcome. "
        "Start fresh with the given information and derive the answer systematically."
    ),
    "Hindsight": (
        "You are rationalizing a guess instead of showing genuine reasoning. "
        "Provide the actual logical steps that led to your conclusion."
    ),
    "Overgeneralization": (
        "You are applying overly broad rules without considering exceptions. "
        "Check if your general principle actually applies to this specific case."
    ),
    "Other": (
        "Your reasoning shows systematic errors that need correction. "
        "Review your approach carefully and verify each logical step."
    ),
    "None": (
        "Your reasoning is sound and your answer is correct. "
        "Good work applying logical thinking to solve the problem."
    )
}


def coaching_from_bias(bias_label: str, context: Optional[str] = None) -> str:
    """
    Convert a bias label to actionable coaching feedback.
    
    Args:
        bias_label: The detected bias (e.g., "Confirmation", "Anchoring")
        context: Optional additional context about the specific error
        
    Returns:
        Coaching feedback string (max 2 sentences, second person)
    """
    # Normalize the bias label
    normalized_label = bias_label.strip()
    
    # Get base coaching feedback
    coaching = BIAS_COACHING.get(normalized_label)
    
    if coaching is None:
        # Fallback for unknown bias labels
        coaching = BIAS_COACHING["Other"]
    
    # If context is provided, we could customize the feedback
    # For now, we'll use the base coaching
    if context and len(context.strip()) > 0:
        # Future enhancement: could incorporate context-specific guidance
        pass
    
    return coaching


def get_available_biases() -> list[str]:
    """Get list of available bias labels."""
    return list(BIAS_COACHING.keys())


def validate_bias_label(bias_label: str) -> bool:
    """Check if a bias label is recognized."""
    return bias_label.strip() in BIAS_COACHING


# Example usage and testing
if __name__ == "__main__":
    # Test the coaching system
    test_biases = ["Confirmation", "Anchoring", "None", "InvalidBias"]
    
    for bias in test_biases:
        coaching = coaching_from_bias(bias)
        print(f"{bias}: {coaching}")
        print()
