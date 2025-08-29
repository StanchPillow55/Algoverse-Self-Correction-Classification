#!/usr/bin/env python3
from __future__ import annotations
import re
from typing import List

def normalize_numeric(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    s = re.sub(r"[ ,]", "", s)
    s = re.sub(r"[^0-9\-\.eE+]", "", s)
    return s

def gsm8k_em(answer: str, reference: str) -> int:
    a = normalize_numeric(answer)
    r = normalize_numeric(reference)
    try:
        if "." in a or "." in r or "e" in a.lower() or "e" in r.lower():
            return int(abs(float(a) - float(r)) < 1e-9)
        return int(int(float(a)) == int(float(r)))
    except Exception:
        return int(a == r)

def humaneval_pass_at_k(passes: List[bool], k: int) -> float:
    n = len(passes)
    if n == 0 or k <= 0:
        return 0.0
    k = min(k, n)
    return float(any(passes[:k]))

