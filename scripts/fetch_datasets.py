#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.dataset_loader import read_csv_flexible
URL1 = "https://github.com/StanchPillow55/Algoverse-Self-Correction-Classification/blob/feat/initial-error-table/data/error_bias_examples_v3.csv"
URL2 = "https://github.com/StanchPillow55/Algoverse-Self-Correction-Classification/blob/feat/initial-error-table/data/ground_truth_qna.csv"
if __name__ == "__main__":
    print("Caching datasets from URLs...")
    try:
        read_csv_flexible(URL1, cache_dir="data/cache")
        print("✅ Cached error bias examples")
        read_csv_flexible(URL2, cache_dir="data/cache")
        print("✅ Cached ground truth QnA")
        print("✅ All datasets cached in data/cache/")
    except Exception as e:
        print(f"⚠️ Failed to cache datasets: {e}")
        exit(1)
