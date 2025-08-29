#!/usr/bin/env python3
"""
Robust HumanEval loader with fallbacks and normalized schema.
Order of resolution:
 1) Local path data/humaneval/HumanEval.jsonl
 2) If missing, clone openai/human-eval and copy data/HumanEval.jsonl
 3) If offline, fallback to built-in demo (3 items)
Normalization:
  { qid, question(prompt), entry_point, test, topic="humaneval" }
"""
import json, os, subprocess, shutil
from pathlib import Path
from typing import List, Dict, Any, Optional

DEFAULT_LOCAL = Path("data/humaneval/HumanEval.jsonl")
DEFAULT_REPO = "https://github.com/openai/human-eval.git"

def _ensure_local_jsonl(local_path: Path = DEFAULT_LOCAL) -> Optional[Path]:
    if local_path.exists():
        return local_path
    tmp = Path(".cache/tmp_he_repo")
    try:
        tmp.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", DEFAULT_REPO, str(tmp)], check=True)
        src = tmp / "data/HumanEval.jsonl"
        if src.exists():
            local_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, local_path)
            return local_path
    except Exception:
        pass
    finally:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
    return None

def _read_jsonl(p: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out

def _normalize(raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    norm: List[Dict[str, Any]] = []
    for i, it in enumerate(raw):
        norm.append({
            "qid": it.get("task_id", f"HumanEval/{i}"),
            "question": it.get("prompt", ""),
            "entry_point": it.get("entry_point", ""),
            "test": it.get("test", ""),
            "topic": "humaneval"
        })
    return norm

def load_humaneval_dataset(subset: str = "full") -> List[Dict[str, Any]]:
    p = _ensure_local_jsonl(DEFAULT_LOCAL)
    if p is not None:
        data = _normalize(_read_jsonl(p))
    else:
        data = create_demo_humaneval_data()
    if subset == "subset_20":
        return data[:20]
    if subset == "subset_100":
        return data[:100]
    return data

def create_demo_humaneval_data() -> List[Dict[str, Any]]:
    """Create minimal demo HumanEval data for testing without API access."""
    return [
        {
            "qid": "HumanEval/0",
            "question": "from typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n",
            "test": "\n\nMETADATA = {\n    'author': 'jt',\n    'dataset': 'test'\n}\n\n\ndef check(candidate):\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False\n    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 0.5) == False\n\n",
            "entry_point": "has_close_elements",
            "topic": "humaneval"
        },
        {
            "qid": "HumanEval/1",
            "question": "from typing import List\n\ndef separate_paren_groups(paren_string: str) -> List[str]:\n    \"\"\" Input to this function is a string containing multiple groups of nested parentheses. Your goal is to\n    separate those group and return the list of those.\n    Separate groups are balanced (each open brace is properly closed) and not nested within each other\n    Ignore any spaces in the input string.\n    >>> separate_paren_groups('( ) (( )) (( )( ))')\n    ['()', '(())', '(()())']\n    \"\"\"\n",
            "test": "\n\nMETADATA = {\n    'author': 'jt',\n    'dataset': 'test'\n}\n\n\ndef check(candidate):\n    assert candidate('(()()) ((())) () ((())()())') == [\n        '(()())', '((()))', '()', '((())()())'\n    ]\n    assert candidate('() (()) ((())) (((())))') == [\n        '()', '(())', '((()))', '(((())))'\n    ]\n    assert candidate('(()(())((())))') == [\n        '(()(())((())))'\n    ]\n    assert candidate('( ) (( )) (( )( ))') == ['()', '(())', '(()())']\n",
            "entry_point": "separate_paren_groups",
            "topic": "humaneval"
        },
        {
            "qid": "HumanEval/2",
            "question": "def truncate_number(number: float) -> float:\n    \"\"\" Given a positive floating point number, it can be decomposed into\n    and integer part (largest integer smaller than given number) and decimals\n    (leftover part always smaller than 1).\n\n    Return the decimal part of the number.\n    >>> truncate_number(3.5)\n    0.5\n    \"\"\"\n",
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
