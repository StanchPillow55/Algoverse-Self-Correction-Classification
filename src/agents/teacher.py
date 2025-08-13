import re
from typing import Dict, Any, List, Tuple

BIAS = ["Anchoring","Confirmation","Availability/Bandwagon","Hindsight","Overgeneralization","None"]

def _contains_any(s: str, toks: List[str]) -> bool:
    return any(tok in s.lower() for tok in toks)

def detect_bias(question: str, answer: str, reference: str, history: List[Dict[str, Any]]) -> Tuple[str, float]:
    """
    Heuristic labeler for 5 biases; returns (bias_label, teacher_confidence)
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
        return "Availability/Bandwagon", 0.7

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
