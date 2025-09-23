#!/usr/bin/env python3
"""
GSM8K loader that handles JSONL format.
"""
import json
import os
from pathlib import Path
from typing import List, Dict, Any

def load_gsm8k_dataset(path: str = "data/gsm8k/test_100.jsonl") -> List[Dict[str, Any]]:
    """Load GSM8K dataset from JSONL or JSON file."""
    # For smoke test, use smaller subset
    if os.getenv("SMOKE_TEST", "0") == "1":
        path = "data/gsm8k/test_100.jsonl"
    elif not Path(path).exists() and Path("data/gsm8k/test_1k.jsonl").exists():
        path = "data/gsm8k/test_1k.jsonl"
    
    p = Path(path)
    if not p.exists():
        # Fallback to any GSM8K CSV
        for csv_path in Path("data").glob("**/*gsm8k*.csv"):
            import pandas as pd
            df = pd.read_csv(csv_path)
            return df.to_dict(orient="records")
        return []
    
    items = []
    
    # Handle both JSONL and JSON formats
    if path.endswith('.json'):
        # JSON array format
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data:
                items.append({
                    "qid": item.get("qid", item.get("id", f"gsm8k_{len(items)}")),
                    "question": item.get("question", ""),
                    "ground_truth": item.get("ground_truth", item.get("answer", "")),
                    "topic": "gsm8k"
                })
    else:
        # JSONL format (one JSON object per line)
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    # Normalize to our expected schema
                    items.append({
                        "qid": item.get("id", f"gsm8k_{len(items)}"),
                        "question": item.get("question", ""),
                        "ground_truth": item.get("answer", ""),
                        "topic": "gsm8k"
                    })
    return items
