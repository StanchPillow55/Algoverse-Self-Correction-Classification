from __future__ import annotations
import re, math
from typing import Dict, Any, Optional, Tuple

# Simple pattern to match numbers
NUM_PAT = re.compile(r"([-+]?\d+(?:\.\d+)?)", re.I)

def _last_number(text: str) -> Optional[str]:
    """Prefer explicit 'answer:' or '####', else take the last numeric token."""
    m = re.search(r"answer\s*[:=]\s*([-+]?\d+(?:\.\d+)?)", text, re.I)
    if m:
        return m.group(1)
    m = re.search(r"####\s*([-+]?\d+(?:\.\d+)?)", text)
    if m:
        return m.group(1)
    all_nums = NUM_PAT.findall(text)
    return all_nums[-1] if all_nums else None

def _to_num(s: str) -> Optional[float]:
    try:
        return float(s.replace(",", ""))
    except Exception:
        return None

class GSM8KEvaluator:
    def extract_answer(self, text: str) -> Tuple[str, str]:
        """
        Return (raw_extracted, reason_tag).
        reason_tag is 'ok' if found, otherwise one of: no_answer, format_error.
        """
        if not isinstance(text, str) or not text.strip():
            return "", "no_answer"
        val = _last_number(text)
        return (val or "", "ok" if val else "no_answer")

    def normalize(self, s: str) -> str:
        return s.strip().replace(",", "")

    def diagnose(self, pred_text: str, gold_text: str) -> str:
        """Lightweight reasoning-aware diagnosis using the full trace."""
        raw, tag = self.extract_answer(pred_text)
        if tag != "ok":
            return "no_answer"
        pnum = _to_num(raw)
        gnum = _to_num(gold_text)
        if pnum is None or gnum is None:
            return "format_error"
        if math.isclose(pnum, gnum, rel_tol=0, abs_tol=0):
            return "correct"
        trace = pred_text.lower()
        if re.search(r"approx(imate|imation)|round(?:ed|ing)", trace):
            return "approximation_error"
        if re.search(r"\b(add|sum|plus|increase)\b.*\b(subtract|minus|decrease)\b", trace) or \
           re.search(r"\bcarried\b|\bborrowed\b", trace):
            return "arithmetic_slip"
        return "logical_flaw"

    def compare(self, pred_text: str, gold_text: str) -> Dict[str, Any]:
        raw, tag = self.extract_answer(pred_text)
        pred_norm = self.normalize(raw) if raw else ""
        gold_norm = self.normalize(gold_text)
        em = float(pred_norm == gold_norm) if pred_norm and gold_norm else 0.0
        diag = "correct" if em == 1.0 else self.diagnose(pred_text, gold_text)
        return {"em": em, "pred": pred_norm, "gold": gold_norm, "diagnosis": diag}
