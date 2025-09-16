import json
from pathlib import Path
from typing import Optional, Tuple

BIAS = ["Anchoring","Confirmation","Availability","Bandwagon","Hindsight","Overgeneralization","Logic-error","None"]

def load_templates(path: str | Path = "rts_templates.json"):
    with open(path, "r", encoding="utf-8") as f:
        return {t["id"]: t for t in json.load(f)}

def bucket_conf(p: float) -> str:
    if p is None: return "mid"
    return "low" if p < 0.4 else ("high" if p > 0.7 else "mid")

def select_template(bias: str, conf: float, is_correct: bool, history_len: int, 
                    is_code_task: bool = False) -> Tuple[bool, str]:
    """
    Returns (reprompt?, template_id)
    
    Args:
        bias: Detected cognitive bias
        conf: Confidence score
        is_correct: Whether the answer was correct
        history_len: Length of conversation history
        is_code_task: Whether this is a code generation task
    """
    if is_correct:
        return (False, None)

    b = bias if bias in BIAS else "None"
    c = bucket_conf(conf)

    # Code-specific template selection
    if is_code_task:
        return select_code_template(b, c)
    
    # Traditional math task template selection
    return select_math_template(b, c)

def select_code_template(bias: str, conf_bucket: str) -> Tuple[bool, str]:
    """
    Select coaching templates specifically for code generation tasks.
    """
    # Code-specific bias handling
    if bias == "Anchoring":
        return True, "counter_anchor_code_v1" if conf_bucket == "high" else "generalize_from_examples_v1"
    
    if bias == "Availability":
        return True, "explore_alternatives_v1" if conf_bucket == "high" else "match_pattern_to_problem_v1"
    
    if bias == "Bandwagon":
        return True, "justify_choices_v1" if conf_bucket == "high" else "simple_over_trendy_v1"
    
    if bias == "Hindsight":
        return True, "test_assumptions_v1" if conf_bucket == "high" else "debug_systematically_v1"
    
    if bias == "Overgeneralization":
        return True, "handle_edge_cases_v1" if conf_bucket == "high" else "flexible_patterns_v1"
    
    if bias == "Logic-error":
        return True, "step_by_step_debug_v1" if conf_bucket == "high" else "verify_requirements_v1"
    
    # Fallback for code tasks
    return True, "step_by_step_debug_v1"

def select_math_template(bias: str, conf_bucket: str) -> Tuple[bool, str]:
    """
    Select coaching templates for traditional math reasoning tasks.
    """
    # Original policy for math tasks
    if bias == "Anchoring" and conf_bucket == "high":
        return True, "counter_anchor_v1"
    
    if bias == "Confirmation" and conf_bucket in ("mid", "high"):
        return True, "devils_advocate_v1"
    
    if bias in ["Availability", "Availability/Bandwagon"] and conf_bucket == "mid":
        return True, "evidence_only_v1"
    
    if bias == "Hindsight":
        return True, "recompute_no_story_v1"
    
    if bias == "Overgeneralization" and conf_bucket in ("low", "mid"):
        return True, "quantify_uncertainty_v1"

    # General fallbacks by confidence
    if conf_bucket == "high":
        return True, "are_you_sure_recheck"
    if conf_bucket == "low":
        return True, "think_step_by_step"
    
    return True, "try_again_concise"
