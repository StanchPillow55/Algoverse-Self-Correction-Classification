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

def extract_boolean_answer(text: str) -> str | None:
    """
    Extract Yes/No answers from text for boolean question datasets like SuperGLUE BoolQ.
    
    Looks for patterns like:
    - Final Answer: Yes/No
    - Answer: Yes/No
    - Therefore: Yes/No
    - Conclusion: Yes/No
    - Last Yes/No in the text
    """
    if not text:
        return None
        
    text = text.strip()
    
    # Priority patterns - look for explicit answer markers
    boolean_patterns = [
        r"(?:final\s+answer|answer|result|conclusion|therefore|thus|so)(?:\s*is)?:?\s*(yes|no)",
        r"(?:the\s+)?answer\s+(?:is\s+)?(yes|no)",
        r"(?:final|conclusion):\s*(yes|no)",
        r"\b(yes|no)\b(?:\s*[.,!]?\s*$)",  # Yes/No at end of text
    ]
    
    # Try patterns in priority order
    for pattern in boolean_patterns:
        matches = list(re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE))
        if matches:
            # Take the last match
            answer = matches[-1].group(1).strip().lower()
            return "Yes" if answer == "yes" else "No"
    
    # Fallback: find all yes/no occurrences and take the last one
    yes_no_matches = list(re.finditer(r"\b(yes|no)\b", text, re.IGNORECASE))
    if yes_no_matches:
        answer = yes_no_matches[-1].group(1).strip().lower()
        return "Yes" if answer == "yes" else "No"
    
    return None

def extract_mathbench_answer(text: str) -> str | None:
    """
    Extract answers from text for MathBench dataset problems.
    
    Handles various answer formats including:
    - Simple numbers: "42", "3.14"
    - Expressions: "x = 4", "2x + 3"
    - Multiple solutions: "x = 2 or x = 3"
    - Complex expressions: "λ₁ = 3, λ₂ = 1"
    - Equations: "y = Ce^(2x)"
    """
    if not text:
        return None
        
    text = text.strip()
    
    # Priority patterns - look for explicit answer markers first
    mathbench_patterns = [
        # Final Answer: [expression]
        r"(?:final\s+answer|answer|result|solution)(?:\s*is)?:?\s*([^\n.]+?)(?:\.|$|\n)",
        # Multiple solutions with "or" - more specific pattern
        r"([a-zA-Z]\s*=\s*[^,\n]+?\s+or\s+[a-zA-Z]\s*=\s*[^,\n.]+?)(?:\.|$|\n)",
        # Mathematical expressions with equals
        r"([a-zA-Z]\s*=\s*[^\n.]+?)(?:\.|$|\n)",
        # Complex expressions with Greek letters or subscripts
        r"([λαβγδεζηθικμνξοπρστυφχψω₁₂₃₄₅₆₇₈₉₀].*?=.*?[^\n.]+?)(?:\.|$|\n)",
        # Mathematical expressions (derivatives, etc.) - look for patterns like "2x + 3"
        r"([+-]?\d*[a-zA-Z]\s*[+-]\s*\d+|\d*[a-zA-Z]\^?\d*\s*[+-]\s*\d*[a-zA-Z]?\s*[+-]?\s*\d*)(?:\s|$)",
        # Simple numeric answers
        r"(?:answer|result|equals?)\s*(?:is\s*)?([+-]?\d+(?:\.\d+)?)",
        # Last meaningful expression on a line
        r"([+-]?\d+(?:\.\d+)?)\s*$"
    ]
    
    # Try patterns in priority order
    for pattern in mathbench_patterns:
        matches = list(re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE))
        if matches:
            # Take the last match
            answer = matches[-1].group(1).strip()
            # Clean up common artifacts
            answer = re.sub(r'^[,.;]\s*', '', answer)  # Remove leading punctuation
            answer = re.sub(r'\s*[,.;]\s*$', '', answer)  # Remove trailing punctuation
            if answer and answer not in ['the', 'is', 'are', 'equals', 'equal']:
                return answer
    
    # Fallback: use the existing numeric extraction for simple cases
    numeric_result = extract_final_answer(text)
    if numeric_result:
        return numeric_result
    
    return None

def extract_toolqa_answer(text: str) -> str | None:
    """
    Extract answers from text for ToolQA dataset problems.
    
    ToolQA has diverse answer formats:
    - Times: "8:00 PM", "21:43", "7:00 PM"
    - Durations: "3:00:00", "1:30:00"
    - Names: "Grace", "Joseph", "Alan"
    - Numbers: "147.0", "5", "-10.0", "$479"
    - Yes/No: "Yes", "No" 
    - Multiple values: "2873, 1176, 1340, 2398"
    - Complex: "nan", "Private room", "Entire home/apt"
    """
    if not text:
        return None
        
    text = text.strip()
    
    # Priority patterns for different ToolQA answer types
    # Order matters - more specific patterns first
    
    # 1. Explicit answer markers (highest priority)
    # Handle "Final Answer:" specifically first - prioritize full expressions
    final_answer_match = re.search(r"(?:final\s+answer)\s*:?\s*([^\n]+?)(?:\.|$|\n)", text, re.IGNORECASE | re.MULTILINE)
    if final_answer_match:
        answer = final_answer_match.group(1).strip()
        answer = re.sub(r'^[,.;:]\s*', '', answer)
        answer = re.sub(r'\s*[,.;:]\s*$', '', answer)
        return answer
    
    # Handle "The answer is X" patterns - extract X precisely
    # First try numeric patterns (with units)
    answer_numeric_with_unit = re.search(r"(?:the\s+answer\s+is|answer\s+is)\s+([+-]?\d+(?:\.\d+)?\s*[A-Z]+)(?:\s|[.,]|$|\n)", text, re.IGNORECASE)
    if answer_numeric_with_unit:
        return answer_numeric_with_unit.group(1).strip()
    
    # Try simple numeric patterns  
    answer_numeric = re.search(r"(?:the\s+answer\s+is|answer\s+is)\s+([+-]?\d+(?:\.\d+)?)(?:\s|[.,]|$|\n)", text, re.IGNORECASE)
    if answer_numeric:
        return answer_numeric.group(1).strip()
    
    # Handle "The answer is Yes/No, ..." separately
    answer_yesno_match = re.search(r"(?:the\s+answer\s+is|answer\s+is)\s+(Yes|No)(?:\s*[.,]|$)", text, re.IGNORECASE)
    if answer_yesno_match:
        return answer_yesno_match.group(1)
        
    # Handle "The answer is [multi-word answer]" - more general pattern
    answer_general = re.search(r"(?:the\s+answer\s+is|answer\s+is)\s+([^\n.]+?)(?:\.|$|\n)", text, re.IGNORECASE)
    if answer_general:
        answer = answer_general.group(1).strip()
        # Clean up common artifacts
        answer = re.sub(r'^[,.;:]\s*', '', answer)
        answer = re.sub(r'\s*[,.;:]\s*$', '', answer)
        return answer
    
    # Handle "I conclude that X" pattern
    conclude_match = re.search(r"I\s+conclude\s+that\s+([^\n.]+?)(?:\.|$|\n)", text, re.IGNORECASE)
    if conclude_match:
        answer = conclude_match.group(1).strip()
        answer = re.sub(r'^[,.;:]\s*', '', answer)
        answer = re.sub(r'\s*[,.;:]\s*$', '', answer)
        return answer
    
    # Other explicit patterns
    other_explicit_patterns = [
        r"(?:answer|result|solution)\s*:?\s*([^\n.]+?)(?:\.|$|\n)",
        r"therefore[,:]*\s*([^\n.]+?)(?:\.|$|\n)",
    ]
    
    for pattern in other_explicit_patterns:
        matches = list(re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE))
        if matches:
            answer = matches[-1].group(1).strip()
            # Clean up
            answer = re.sub(r'^[,.;:]\s*', '', answer)
            answer = re.sub(r'\s*[,.;:]\s*$', '', answer)
            if answer and answer.lower() not in ['the', 'is', 'are', 'was', 'were', 'a', 'an']:
                return answer
    
    # 2. Duration formats (full HH:MM:SS)
    duration_match = re.search(r'\b(\d+:\d{2}:\d{2})\b', text)
    if duration_match:
        return duration_match.group(1)
    
    # 3. Time formats with AM/PM
    time_ampm_match = re.search(r'\b(\d{1,2}:\d{2}\s*(?:AM|PM))\b', text, re.IGNORECASE)
    if time_ampm_match:
        return time_ampm_match.group(1)
    
    # 4. 24-hour time formats
    time_24h_match = re.search(r'\b(\d{1,2}:\d{2})\b', text)
    if time_24h_match:
        return time_24h_match.group(1)
    
    # 5. Yes/No answers
    yesno_patterns = [
        r'\b(Yes|No)\b(?:\s*[.,!]?\s*$)',  # Yes/No at end
        r'\b(Yes|No)\b',  # Any Yes/No
    ]
    for pattern in yesno_patterns:
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        if matches:
            answer = matches[-1].group(1)
            return "Yes" if answer.lower() == "yes" else "No"
    
    # 6. Names appearing before other patterns  
    # Look for names in context of "[Name] attended" or "is [Name]" or "was [Name]"
    name_context_patterns = [
        r'\b([A-Z][a-z]+)\s+attended\b',  # "Grace attended"
        r'\bis\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b',
        r'\bwas\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b',
        r'\battended\s+(?:by\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b',
        r'\bhost\s+(?:is\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b',
        r'\b([A-Z][a-z]+)\s+(?:was|is)\s+the\b',  # "Grace was the..."
    ]
    
    for pattern in name_context_patterns:
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        if matches:
            name = matches[-1].group(1).strip()
            # Filter out common non-names
            if name.lower() not in ['the', 'this', 'that', 'event', 'meeting', 'time', 'answer', 'question', 'delay', 'flight']:
                return name
    
    # 7. Multiple comma-separated numbers
    multi_numbers_match = re.search(r'\b(\d+(?:,\s*\d+)+)\b', text)
    if multi_numbers_match:
        return multi_numbers_match.group(1)
    
    # 7. Special room types
    room_match = re.search(r'\b(Private room|Entire home/apt|Shared room)\b', text, re.IGNORECASE)
    if room_match:
        return room_match.group(1)
    
    # 8. Price formats
    price_match = re.search(r'\$(\d+(?:,\d+)?(?:\.\d{2})?)', text)
    if price_match:
        return '$' + price_match.group(1)
    
    # 9. Decimal numbers (preserve full precision) including negatives
    # Look for negative numbers that might be preceded by "was " 
    negative_decimal_match = re.search(r'\bwas\s+([+-]\d+\.\d+)\b', text)
    if negative_decimal_match:
        return negative_decimal_match.group(1)
    
    decimal_match = re.search(r'\b([+-]?\d+\.\d+)\b', text)
    if decimal_match:
        return decimal_match.group(1)
    
    # 10. Whole numbers (including negatives)
    negative_number_match = re.search(r'\bwas\s+([+-]\d+)\b', text)
    if negative_number_match:
        return negative_number_match.group(1)
        
    number_match = re.search(r'\b([+-]?\d+)\b', text)
    if number_match:
        return number_match.group(1)
    
    # 11. Special values
    if 'nan' in text.lower():
        return 'nan'
    
    # 13. Quoted content
    quote_match = re.search(r'"([^"]+)"', text)
    if quote_match:
        return quote_match.group(1)
    
    return None

def humaneval_pass_at_k(passes: List[bool], k: int) -> float:
    n = len(passes)
    if n == 0 or k <= 0:
        return 0.0
    k = min(k, n)
    return float(any(passes[:k]))

