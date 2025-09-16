#!/usr/bin/env python3
"""
Code-Specific Coaching Templates

This module provides targeted coaching templates for code generation tasks,
designed to address specific cognitive biases identified in programming responses.

Each template provides actionable guidance tailored to common programming mistakes
and cognitive biases that affect code quality.
"""

from typing import Dict, Any, Optional


CODE_COACHING_TEMPLATES = {
    # Anchoring Bias Templates
    "counter_anchor_code_v1": {
        "prompt": """Don't hardcode values from the examples. Step back and identify the general pattern or algorithm needed. 

Your previous solution appears to anchor on specific values from the problem examples. Instead:

1. Ignore the specific numbers/strings in the examples
2. Focus on the underlying logic and relationships
3. Write code that works for ANY valid input, not just the given examples
4. Test your mental model with different inputs

What general pattern or algorithm does this problem require, regardless of the specific example values?""",
        "context": "anchoring_bias_code"
    },

    "generalize_from_examples_v1": {
        "prompt": """Look beyond the specific examples to find the general solution pattern.

The examples are just illustrations - your code needs to handle the entire problem space:

1. What is the core logic, independent of example values?
2. What would change if the input values were completely different?
3. Are you making assumptions based only on the given examples?

Rewrite your solution to be truly general, not tied to the specific examples shown.""",
        "context": "anchoring_bias_code"
    },

    # Availability Heuristic Templates  
    "explore_alternatives_v1": {
        "prompt": """Consider alternative approaches beyond your first instinct. 

Your solution uses familiar patterns that might not be optimal for this specific problem:

1. What other algorithms or data structures could solve this?
2. Is there a more direct approach than your current method?
3. Are you using a pattern because it's familiar, or because it's the best fit?

Challenge yourself to think of at least 2 different approaches before settling on one.""",
        "context": "availability_heuristic_code"
    },

    "match_pattern_to_problem_v1": {
        "prompt": """Make sure your chosen approach actually fits this specific problem.

Rather than defaulting to familiar patterns, consider:

1. What are the unique characteristics of THIS problem?
2. What approach is most natural for these specific constraints?
3. Are you overcomplicating with familiar but unnecessary patterns?

Choose the approach that best matches the problem structure, not just what comes to mind first.""",
        "context": "availability_heuristic_code"
    },

    # Bandwagon Effect Templates
    "justify_choices_v1": {
        "prompt": """Explain why you chose this specific approach for THIS problem.

Your solution uses popular/trendy patterns without clear justification:

1. Why is this approach better than alternatives for this specific problem?
2. What advantages does it provide in this context?
3. Are you using it because it's popular, or because it's the right tool?

Don't just use popular patterns—use the RIGHT patterns. Justify your technical choices.""",
        "context": "bandwagon_effect_code"
    },

    "simple_over_trendy_v1": {
        "prompt": """Prioritize clarity and simplicity over trendy language features.

Consider whether simpler approaches might be more appropriate:

1. Could this be solved more clearly with basic constructs?
2. Are you adding complexity for its own sake?
3. Would a junior developer easily understand your approach?

Sometimes the most elegant solution is the simplest one, even if it's not the most "modern".""",
        "context": "bandwagon_effect_code"
    },

    # Hindsight Bias Templates
    "test_assumptions_v1": {
        "prompt": """Test your assumptions systematically rather than being overconfident.

Your reasoning showed high confidence, but the solution failed:

1. What assumptions did you make that might be wrong?
2. Run through edge cases step by step
3. What could go wrong with your current approach?
4. Where might your logic break down?

Approach the problem with healthy skepticism about your initial solution.""",
        "context": "hindsight_bias_code"
    },

    "debug_systematically_v1": {
        "prompt": """Debug your approach step by step instead of assuming it should work.

Rather than being confident about correctness:

1. Trace through your code with the failing test case
2. Identify exactly where the logic breaks down
3. Question each step - why should this work?
4. Look for off-by-one errors, boundary conditions, and edge cases

Be methodical in finding the actual source of the problem.""",
        "context": "hindsight_bias_code"
    },

    # Overgeneralization Templates
    "handle_edge_cases_v1": {
        "prompt": """Consider edge cases and exceptions to make your solution robust.

Your code appears to make rigid assumptions:

1. What happens with empty inputs or single elements?
2. How does your solution handle boundary values?
3. Are there special cases that need different handling?
4. What assumptions might not hold for all valid inputs?

Make your solution flexible enough to handle the full range of possible inputs.""",
        "context": "overgeneralization_code"
    },

    "flexible_patterns_v1": {
        "prompt": """Use flexible patterns that adapt to different scenarios.

Avoid overly rigid code structures:

1. Can your algorithm adapt to different input sizes/types?
2. Are you making unnecessary assumptions about the data?
3. Would your approach work if the problem constraints were slightly different?

Build adaptability into your solution rather than coding for just one scenario.""",
        "context": "overgeneralization_code"
    },

    # Logic Error Templates (Fallback)
    "step_by_step_debug_v1": {
        "prompt": """Debug your logic step by step to find the error.

Your code has a logical error but no clear bias pattern:

1. Trace through your algorithm with a simple example
2. Check each conditional and loop carefully
3. Verify your understanding of the problem requirements
4. Look for common mistakes: off-by-one errors, wrong comparisons, missing cases

Work through the logic methodically to identify where it goes wrong.""",
        "context": "logic_error_code"
    },

    "verify_requirements_v1": {
        "prompt": """Double-check that you understand the problem requirements correctly.

Review the problem statement carefully:

1. Are you solving exactly what's being asked?
2. Did you miss any constraints or requirements?
3. Are you returning the right format/type?
4. Do your variable names and logic match the problem description?

Sometimes the issue is a misunderstanding of what needs to be solved.""",
        "context": "logic_error_code"
    }
}


def get_code_coaching_for_bias(bias_label: str, confidence: float = 0.5) -> Dict[str, Any]:
    """
    Get appropriate coaching template for detected bias in code generation.
    
    Args:
        bias_label: The type of cognitive bias detected
        confidence: Confidence score for the bias detection
        
    Returns:
        Dictionary containing template information
    """
    # Map bias types to preferred templates
    bias_to_template = {
        "Anchoring": "counter_anchor_code_v1" if confidence > 0.6 else "generalize_from_examples_v1",
        "Availability": "explore_alternatives_v1" if confidence > 0.6 else "match_pattern_to_problem_v1", 
        "Bandwagon": "justify_choices_v1" if confidence > 0.6 else "simple_over_trendy_v1",
        "Hindsight": "test_assumptions_v1" if confidence > 0.6 else "debug_systematically_v1",
        "Overgeneralization": "handle_edge_cases_v1" if confidence > 0.6 else "flexible_patterns_v1",
        "Logic-error": "step_by_step_debug_v1" if confidence > 0.6 else "verify_requirements_v1"
    }
    
    template_id = bias_to_template.get(bias_label, "step_by_step_debug_v1")
    template = CODE_COACHING_TEMPLATES.get(template_id, CODE_COACHING_TEMPLATES["step_by_step_debug_v1"])
    
    return {
        "template_id": template_id,
        "prompt": template["prompt"],
        "context": template["context"],
        "bias_detected": bias_label,
        "confidence": confidence
    }


def get_coaching_explanation(bias_label: str, confidence: float) -> str:
    """
    Get explanation of why this coaching was selected.
    
    Args:
        bias_label: The detected cognitive bias
        confidence: Confidence in the bias detection
        
    Returns:
        Human-readable explanation
    """
    explanations = {
        "Anchoring": "Coaching focuses on moving beyond specific example values to general patterns",
        "Availability": "Coaching encourages exploring alternatives beyond familiar first approaches",
        "Bandwagon": "Coaching emphasizes justifying technical choices rather than following trends",
        "Hindsight": "Coaching promotes systematic testing rather than overconfident assumptions",
        "Overgeneralization": "Coaching guides toward flexible solutions that handle edge cases",
        "Logic-error": "Coaching provides systematic debugging approaches for logical errors"
    }
    
    base_explanation = explanations.get(bias_label, "General debugging guidance provided")
    return f"{base_explanation} (bias confidence: {confidence:.2f})"


def format_coaching_prompt(template_info: Dict[str, Any], question: str, previous_answer: str) -> str:
    """
    Format the coaching prompt with context from the specific problem.
    
    Args:
        template_info: Template information from get_code_coaching_for_bias
        question: The original problem statement
        previous_answer: The previous (incorrect) solution attempt
        
    Returns:
        Formatted coaching prompt ready to send to the learner
    """
    base_prompt = template_info["prompt"]
    
    # Add context about the specific problem
    context_header = f"""Based on your previous solution to this problem, here's some targeted feedback:

PROBLEM: {question[:200]}{'...' if len(question) > 200 else ''}

"""
    
    # Add footer encouraging systematic approach
    context_footer = """

Take a moment to think through this systematically before writing new code."""
    
    return context_header + base_prompt + context_footer