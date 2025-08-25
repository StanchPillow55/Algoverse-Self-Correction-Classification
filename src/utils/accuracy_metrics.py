import re
import ast
from typing import Dict, Any, Tuple
from difflib import SequenceMatcher

def normalize_code(code: str) -> str:
    """Normalize code by removing extra whitespace and comments"""
    if not code:
        return ""
    
    # Remove comments
    code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
    
    # Normalize whitespace
    code = re.sub(r'\s+', ' ', code)
    code = code.strip()
    
    # Remove empty lines
    lines = [line.strip() for line in code.split('\n') if line.strip()]
    return '\n'.join(lines)

def code_similarity(answer: str, reference: str) -> float:
    """Calculate similarity between two code snippets"""
    if not answer or not reference:
        return 0.0
    
    norm_answer = normalize_code(answer)
    norm_reference = normalize_code(reference)
    
    # Exact match after normalization
    if norm_answer == norm_reference:
        return 1.0
    
    # Sequence similarity
    similarity = SequenceMatcher(None, norm_answer, norm_reference).ratio()
    
    # Boost similarity if they're functionally equivalent
    try:
        # Try to parse both as Python code
        ast.parse(norm_answer)
        ast.parse(norm_reference)
        # If both parse successfully, they might be functionally equivalent
        if similarity > 0.7:  # High similarity threshold
            similarity += 0.2
    except SyntaxError:
        pass
    
    return min(1.0, similarity)

def math_accuracy(answer: str, reference: str) -> float:
    """Calculate accuracy for mathematical problems"""
    try:
        if "." in answer or "." in reference:
            return int(abs(float(answer) - float(reference)) < 1e-9)
        return int(int(float(answer)) == int(float(reference)))
    except Exception:
        return 0.0

def text_similarity(answer: str, reference: str) -> float:
    """Calculate similarity for text-based problems"""
    if not answer or not reference:
        return 0.0
    
    # Exact match
    if answer.strip() == reference.strip():
        return 1.0
    
    # Sequence similarity
    return SequenceMatcher(None, answer.strip(), reference.strip()).ratio()

def get_accuracy(answer: str, reference: str, dataset_type: str = "auto") -> Tuple[float, Dict[str, Any]]:
    """
    Calculate accuracy based on dataset type.
    
    Args:
        answer: Model's answer
        reference: Ground truth reference
        dataset_type: "math", "code", "text", or "auto"
    
    Returns:
        (accuracy_score, metadata)
    """
    if dataset_type == "auto":
        # Auto-detect based on content
        if any(char in reference for char in "def class import from"):
            dataset_type = "code"
        elif any(char in reference for char in "0123456789+-*/"):
            dataset_type = "math"
        else:
            dataset_type = "text"
    
    metadata = {
        "dataset_type": dataset_type,
        "raw_answer": answer,
        "raw_reference": reference
    }
    
    if dataset_type == "math":
        acc = math_accuracy(answer, reference)
        metadata["math_exact"] = bool(acc)
    elif dataset_type == "code":
        acc = code_similarity(answer, reference)
        metadata["code_normalized"] = normalize_code(answer)
        metadata["code_similarity"] = acc
    else:  # text
        acc = text_similarity(answer, reference)
        metadata["text_similarity"] = acc
    
    metadata["final_accuracy"] = acc
    return acc, metadata