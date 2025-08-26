#!/usr/bin/env python3
"""
HumanEval dataset loader for code generation tasks.
Loads JSONL data and normalizes to internal schema.
"""
import json
import io
import hashlib
import pathlib
from typing import List, Dict, Any, Optional, Union
try:
    import requests
except ImportError:
    requests = None


def load_humaneval_jsonl(
    src: Union[str, pathlib.Path], 
    cache_dir: Optional[str] = "data/cache"
) -> List[Dict[str, Any]]:
    """
    Load HumanEval dataset from local JSONL file or GitHub URL.
    
    Args:
        src: Local path or GitHub raw URL to JSONL file
        cache_dir: Directory to cache downloaded files
    
    Returns:
        List of dictionaries with HumanEval task data
    """
    if isinstance(src, pathlib.Path):
        src = str(src)
    
    # Handle local files
    if not src.startswith("http"):
        with open(src, 'r', encoding='utf-8') as f:
            return [json.loads(line.strip()) for line in f if line.strip()]
    
    # Handle URLs with caching
    cache_path = None
    if cache_dir:
        h = hashlib.sha1(src.encode()).hexdigest()[:16]
        cache_path = pathlib.Path(cache_dir) / f"humaneval_{h}.jsonl"
        if cache_path.exists():
            with open(cache_path, 'r', encoding='utf-8') as f:
                return [json.loads(line.strip()) for line in f if line.strip()]
    
    # Download and parse
    if requests is None:
        raise ImportError("'requests' is required for URL loading")
    
    response = requests.get(src, timeout=30)
    response.raise_for_status()
    
    data = [json.loads(line.strip()) for line in response.text.split('\n') if line.strip()]
    
    # Cache the result
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
    
    return data


def normalize_humaneval_schema(raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalize HumanEval data to internal schema.
    
    Expected HumanEval format:
    - task_id: unique identifier (e.g., "HumanEval/0")
    - prompt: function signature + docstring
    - canonical_solution: reference implementation
    - test: unit tests to validate solution
    - entry_point: function name to implement
    
    Internal schema:
    - qid: question/task ID
    - question: problem text/prompt
    - reference: reference solution
    - test: test cases
    - entry_point: function name
    - topic: "humaneval"
    """
    normalized = []
    for item in raw_data:
        try:
            normalized_item = {
                "qid": item.get("task_id", f"humaneval_{len(normalized)}"),
                "question": item.get("prompt", ""),
                "reference": item.get("canonical_solution", ""),
                "test": item.get("test", ""),
                "entry_point": item.get("entry_point", ""),
                "topic": "humaneval"
            }
            normalized.append(normalized_item)
        except Exception as e:
            print(f"Warning: Failed to normalize item {item.get('task_id', 'unknown')}: {e}")
            continue
    
    return normalized


def get_humaneval_subset(data: List[Dict[str, Any]], subset_name: str) -> List[Dict[str, Any]]:
    """
    Get a subset of HumanEval data for testing/validation.
    
    Args:
        data: Full HumanEval dataset
        subset_name: One of "subset_20", "subset_100", "full"
    
    Returns:
        Subset of the data
    """
    if subset_name == "full":
        return data
    elif subset_name == "subset_20":
        return data[:20]
    elif subset_name == "subset_100":
        return data[:100]
    else:
        raise ValueError(f"Unknown subset: {subset_name}. Use 'subset_20', 'subset_100', or 'full'")


def load_humaneval_dataset(
    src: str = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl",
    subset: str = "full",
    cache_dir: Optional[str] = "data/cache"
) -> List[Dict[str, Any]]:
    """
    Complete HumanEval loading pipeline.
    
    Args:
        src: Source path or URL to HumanEval JSONL
        subset: Subset to return ("subset_20", "subset_100", "full")
        cache_dir: Cache directory for downloads
    
    Returns:
        Normalized HumanEval data subset
    """
    # Load raw data
    raw_data = load_humaneval_jsonl(src, cache_dir)
    
    # Normalize schema
    normalized_data = normalize_humaneval_schema(raw_data)
    
    # Get subset
    subset_data = get_humaneval_subset(normalized_data, subset)
    
    return subset_data


# For testing/demo purposes, create minimal sample data
def create_demo_humaneval_data() -> List[Dict[str, Any]]:
    """Create minimal demo HumanEval data for testing without API access."""
    return [
        {
            "qid": "HumanEval/0",
            "question": "from typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n",
            "reference": "    for idx, elem in enumerate(numbers):\n        for idx2, elem2 in enumerate(numbers):\n            if idx != idx2:\n                distance = abs(elem - elem2)\n                if distance < threshold:\n                    return True\n\n    return False\n",
            "test": "\n\nMETADATA = {\n    'author': 'jt',\n    'dataset': 'test'\n}\n\n\ndef check(candidate):\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False\n    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 0.5) == False\n\n",
            "entry_point": "has_close_elements",
            "topic": "humaneval"
        },
        {
            "qid": "HumanEval/1", 
            "question": "from typing import List\n\ndef separate_paren_groups(paren_string: str) -> List[str]:\n    \"\"\" Input to this function is a string containing multiple groups of nested parentheses. Your goal is to\n    separate those group and return the list of those.\n    Separate groups are balanced (each open brace is properly closed) and not nested within each other\n    Ignore any spaces in the input string.\n    >>> separate_paren_groups('( ) (( )) (( )( ))')\n    ['()', '(())', '(()())']\n    \"\"\"\n",
            "reference": "    result = []\n    current_string = []\n    current_depth = 0\n\n    for c in paren_string:\n        if c == '(':\n            current_depth += 1\n            current_string.append(c)\n        elif c == ')':\n            current_depth -= 1\n            current_string.append(c)\n\n            if current_depth == 0:\n                result.append(''.join(current_string))\n                current_string = []\n\n    return result\n",
            "test": "\n\nMETADATA = {\n    'author': 'jt',\n    'dataset': 'test'\n}\n\n\ndef check(candidate):\n    assert candidate('(()()) ((())) () ((())()())') == [\n        '(()())', '((()))', '()', '((())()())'\n    ]\n    assert candidate('() (()) ((())) (((())))') == [\n        '()', '(())', '((()))', '(((())))'\n    ]\n    assert candidate('(()(())((())))') == [\n        '(()(())((())))'\n    ]\n    assert candidate('( ) (( )) (( )( ))') == ['()', '(())', '(()())']\n",
            "entry_point": "separate_paren_groups",
            "topic": "humaneval"
        },
        {
            "qid": "HumanEval/2",
            "question": "def truncate_number(number: float) -> float:\n    \"\"\" Given a positive floating point number, it can be decomposed into\n    and integer part (largest integer smaller than given number) and decimals\n    (leftover part always smaller than 1).\n\n    Return the decimal part of the number.\n    >>> truncate_number(3.5)\n    0.5\n    \"\"\"\n",
            "reference": "    return number % 1.0\n",
            "test": "\n\nMETADATA = {\n    'author': 'jt',\n    'dataset': 'test'\n}\n\n\ndef check(candidate):\n    assert candidate(3.5) == 0.5\n    assert abs(candidate(1.33) - 0.33) < 1e-6\n    assert abs(candidate(123.456) - 0.456) < 1e-6\n",
            "entry_point": "truncate_number",
            "topic": "humaneval"
        }
    ]


if __name__ == "__main__":
    # Test the loader
    print("Testing HumanEval loader...")
    try:
        # Try to load from official source (requires internet)
        data = load_humaneval_dataset(subset="subset_20")
        print(f"Loaded {len(data)} items from official HumanEval")
        print(f"First item: {data[0]['qid']}")
    except Exception as e:
        print(f"Failed to load from official source: {e}")
        print("Using demo data...")
        data = create_demo_humaneval_data()
        print(f"Created {len(data)} demo items")
        print(f"First item: {data[0]['qid']}")
