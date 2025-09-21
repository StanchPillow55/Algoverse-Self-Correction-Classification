import re
from typing import Dict, Any, List, Tuple, Optional
from .code_bias_detector import CodeBiasDetector

BIAS = ["Anchoring","Confirmation","Availability","Bandwagon","Hindsight","Overgeneralization","Logic-error","None"]

# Initialize the enhanced code bias detector
_code_bias_detector = CodeBiasDetector()

def _contains_any(s: str, toks: List[str]) -> bool:
    return any(tok in s.lower() for tok in toks)

def detect_bias(question: str, answer: str, reference: str, history: List[Dict[str, Any]], 
                reasoning_text: Optional[str] = None, execution_result: Optional[Dict] = None,
                is_code_task: bool = False) -> Tuple[str, float]:
    """
    Enhanced bias detection supporting both traditional math and code generation tasks.
    
    Args:
        question: The problem statement
        answer: The extracted answer/solution
        reference: The reference answer (may be empty for code tasks)
        history: Previous turn history
        reasoning_text: Full reasoning trace (for code tasks)
        execution_result: Code execution results (for code tasks)
        is_code_task: Whether this is a code generation task
    
    Returns:
        Tuple of (bias_label, confidence_score)
    """
    # Use enhanced detection for code tasks
    if is_code_task and reasoning_text is not None and execution_result is not None:
        return _code_bias_detector.detect_code_bias(
            question, answer, reasoning_text, execution_result, history
        )
    
    # Fall back to original logic for math tasks
    return _detect_bias_original(question, answer, reference, history)

def _detect_bias_original(question: str, answer: str, reference: str, history: List[Dict[str, Any]]) -> Tuple[str, float]:
    """
    Original heuristic labeler for math tasks; returns (bias_label, teacher_confidence)
    """
    ans = (answer or "").strip()
    ref = (reference or "").strip()

    if ans == ref:
        return "None", 0.95

    # Anchoring: parroting numbers from prompt
    nums_q = re.findall(r"\d+", question or "")
    if ans in nums_q:
        return "Anchoring", 0.7

    # Availability/Bandwagon cues
    if _contains_any(ans, ["everyone", "most people", "commonly", "popular"]):
        return "Availability", 0.7

    # Hindsight rationalization cues
    if _contains_any(ans, ["obvious because", "clearly since", "as expected"]) or ("because" in ans and ref and ans != ref):
        return "Hindsight", 0.65

    # Overgeneralization markers
    if _contains_any(ans, ["always", "never", "all cases"]) and ans != ref:
        return "Overgeneralization", 0.65

    # Default: Confirmation (sticking with first guess)
    return "Confirmation", 0.6

def combine_confidence(learner_self_conf: float, teacher_conf: float, k_vote_share: float | None = None) -> float:
    parts = [p for p in [learner_self_conf, teacher_conf, k_vote_share] if p is not None]
    return sum(parts)/len(parts) if parts else 0.5
