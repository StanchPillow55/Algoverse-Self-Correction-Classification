#!/usr/bin/env python3
from __future__ import annotations
import re
from typing import List
from decimal import Decimal, InvalidOperation, getcontext
from fractions import Fraction

getcontext().prec = 50

# Robust regex patterns for number extraction
NUM_RE = r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
AFTER_HASHES_RE = re.compile(r"####\s*(" + NUM_RE + r")")
NUMBER_SCAN_RE = re.compile(NUM_RE)

def _clean_commas(s: str) -> str:
    """Remove commas from numeric strings."""
    return s.replace(",", "").strip()

def _as_fraction(s: str):
    """
    Parse a numeric-like string into a Fraction when possible.
    Handles integers, decimals, and simple forms like '-3/2' or '3 1/2'.
    """
    s = s.strip()
    s = s.replace(",", "")
    # mixed fraction: "a b/c"
    m = re.fullmatch(r"([-+]?\d+)\s+(\d+)/(\d+)", s)
    if m:
        a, b, c = int(m.group(1)), int(m.group(2)), int(m.group(3))
        sign = -1 if a < 0 else 1
        return Fraction(a, 1) + sign*Fraction(b, c)
    # plain fraction a/b
    m = re.fullmatch(r"([-+]?\d+)/(\d+)", s)
    if m:
        return Fraction(int(m.group(1)), int(m.group(2)))
    # decimal or int
    try:
        d = Decimal(s)
        return Fraction(d)  # exact for finite decimals
    except (InvalidOperation, ValueError):
        return None

def normalize_numeric_string(s: str) -> str:
    """
    Canonicalize numeric strings so equivalent values compare equal.
    - strip commas
    - convert decimals/fractions to simplest exact form when finite
    - prefer integer form if denominator is 1
    """
    s = s.strip().replace("%","").replace("$","")
    s = _clean_commas(s)
    f = _as_fraction(s)
    if f is None:
        # as last resort, return cleaned raw
        return s
    if f.denominator == 1:
        return str(f.numerator)
    # finite decimal? represent with Decimal to normalized string
    try:
        d = Decimal(f.numerator) / Decimal(f.denominator)
        # remove trailing zeros
        s = format(d.normalize(), "f").rstrip("0").rstrip(".") if "." in str(d) else str(d)
        return s
    except Exception:
        return f"{f.numerator}/{f.denominator}"

def extract_final_answer(text: str) -> str | None:
    """
    Priority 1: take the number immediately after the last '####'.
    Priority 2: take the last numeric token in the text.
    """
    if not text:
        return None
    # take the last occurrence after #### if present
    mlist = list(AFTER_HASHES_RE.finditer(text))
    if mlist:
        return normalize_numeric_string(mlist[-1].group(1))
    # else last numeric token
    nums = NUMBER_SCAN_RE.findall(text)
    if nums:
        return normalize_numeric_string(nums[-1])
    return None

def gsm8k_extract_gold_answer(reference: str) -> str:
    """
    Extract the final answer from GSM8K reference text.
    Looks for #### followed by the answer.
    """
    if not isinstance(reference, str):
        reference = str(reference)
    m = AFTER_HASHES_RE.search(reference)
    if m:
        return normalize_numeric_string(m.group(1))
    # fallback: last number in reference
    nums = NUMBER_SCAN_RE.findall(reference)
    return normalize_numeric_string(nums[-1]) if nums else reference.strip()

def gsm8k_em(answer: str, reference: str) -> int:
    """
    Improved GSM8K exact match with proper answer extraction and normalization.
    """
    if answer is None or reference is None:
        return 0
    
    # Extract final answers
    pred_final = extract_final_answer(str(answer)) if isinstance(answer, str) else normalize_numeric_string(str(answer))
    gold_final = gsm8k_extract_gold_answer(str(reference))
    
    if pred_final is None or gold_final is None:
        return 0
    
    # Compare normalized answers
    return int(pred_final == gold_final)

# Legacy function for backwards compatibility
def normalize_numeric(s: str) -> str:
    """Legacy function - use extract_final_answer instead."""
    return extract_final_answer(s) or ""

def humaneval_pass_at_k(passes: List[bool], k: int) -> float:
    n = len(passes)
    if n == 0 or k <= 0:
        return 0.0
    k = min(k, n)
    return float(any(passes[:k]))

