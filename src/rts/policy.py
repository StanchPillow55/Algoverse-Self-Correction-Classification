import json
from pathlib import Path

BIAS = ["Anchoring","Confirmation","Availability/Bandwagon","Hindsight","Overgeneralization","None"]

def load_templates(path: str | Path = "rts_templates.json"):
    with open(path, "r", encoding="utf-8") as f:
        return {t["id"]: t for t in json.load(f)}

def bucket_conf(p: float) -> str:
    if p is None: return "mid"
    return "low" if p < 0.4 else ("high" if p > 0.7 else "mid")

def select_template(bias: str, conf: float, is_correct: bool, history_len: int) -> tuple[bool, str | None]:
    """
    Returns (reprompt?, template_id)
    """
    if is_correct:
        return (False, None)

    b = bias if bias in BIAS else "None"
    c = bucket_conf(conf)

    # Quick-start policy
    if b == "Anchoring" and c == "high":       return True, "counter_anchor_v1"
    if b == "Confirmation" and c in ("mid","high"): return True, "devils_advocate_v1"
    if b == "Availability/Bandwagon" and c == "mid": return True, "evidence_only_v1"
    if b == "Hindsight":                        return True, "recompute_no_story_v1"
    if b == "Overgeneralization" and c in ("low","mid"): return True, "quantify_uncertainty_v1"

    # general fallbacks by confidence
    if c == "high":   return True, "are_you_sure_recheck"
    if c == "low":    return True, "think_step_by_step"
    return True, "try_again_concise"
