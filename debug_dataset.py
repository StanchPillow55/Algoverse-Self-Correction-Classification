#!/usr/bin/env python3

import sys
sys.path.append('.')

from src.loop.runner import _load_dataset

# Test loading our custom dataset
print("Testing dataset loading...")
dataset_path = "test_gsm8k_5.json"
print(f"Loading dataset from: {dataset_path}")

try:
    data = _load_dataset(dataset_path)
    print(f"Loaded {len(data)} samples")
    for i, sample in enumerate(data[:3]):  # Show first 3
        print(f"Sample {i+1}: {sample}")
except Exception as e:
    print(f"Error loading dataset: {e}")
    import traceback
    traceback.print_exc()