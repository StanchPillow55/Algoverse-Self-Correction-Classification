#!/usr/bin/env python3
import argparse, pathlib, yaml
from src.utils.dataset_loader import read_csv_flexible
if __name__ == "__main__":
    with open("configs/experiments/datasets.yaml", "r") as f:
        cfg = yaml.safe_load(f)["datasets"]
    print("Fetching and caching datasets...")
    read_csv_flexible(cfg["error_lib_url"], cache_dir="data/cache")
    read_csv_flexible(cfg["qna_math_url"], cache_dir="data/cache")
    print("✅ Datasets cached in data/cache/.")
