#!/usr/bin/env python3
"""
Dataset availability and counts checker.
- Prefer local ./data for HumanEval and GSM8K
- Else try HuggingFace datasets (if internet + datasets pkg available)
- Print counts and exit 0 on success, non-zero on failure.
"""
import os, sys
from pathlib import Path

HE_COUNT_EXPECTED = 164
GSM8K_MIN, GSM8K_MAX = 1000, 2000


def has_internet(url: str = "https://huggingface.co") -> bool:
    try:
        import urllib.request

        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=5) as _:
            return True
    except Exception:
        return False


def human_eval_count() -> int:
    try:
        from src.data.humaneval_loader import load_humaneval_dataset

        return len(load_humaneval_dataset(subset="full"))
    except Exception:
        # Fallback: try local jsonl directly
        p = Path("data/humaneval/HumanEval.jsonl")
        if p.exists():
            return sum(1 for _ in p.open("r", encoding="utf-8"))
        # Fallback to demo => not acceptable for counts
        return -1


def gsm8k_count() -> int:
    # Check for our standardized test_1k.jsonl first
    test_1k = Path("data/gsm8k/test_1k.jsonl")
    if test_1k.exists():
        return sum(1 for _ in test_1k.open("r", encoding="utf-8"))
    
    # Prefer local csv under data/gsm8k*.csv
    for p in Path("data").glob("**/*gsm8k*.csv"):
        try:
            import pandas as pd

            return len(pd.read_csv(p))
        except Exception:
            pass

    # Try HuggingFace if available and internet
    if has_internet() and os.getenv("DISABLE_HF", "0") != "1":
        try:
            from datasets import load_dataset  # type: ignore

            ds = load_dataset("gsm8k", "main")
            # Use test split for evaluation count
            if "test" in ds:
                return len(ds["test"])
            # Some configs expose different splits; consider union
            return sum(len(ds[k]) for k in ds.keys() if isinstance(ds[k], type(ds["train"])) )
        except Exception:
            pass

    return -1


def main() -> int:
    he = human_eval_count()
    gsm = gsm8k_count()

    # Print counts for CI logs
    print(f"HumanEval_count={he}")
    print(f"GSM8K_count={gsm}")

    # Validate
    ok_he = he == HE_COUNT_EXPECTED
    ok_gsm = GSM8K_MIN <= gsm <= GSM8K_MAX

    if not ok_he:
        print(f"ERROR: HumanEval count {he} != {HE_COUNT_EXPECTED}")
        return 2
    if not ok_gsm:
        print(f"ERROR: GSM8K count {gsm} not in [{GSM8K_MIN},{GSM8K_MAX}]")
        return 3
    print("OK: dataset availability checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

