#!/usr/bin/env python3
import argparse, pathlib, pandas as pd
from src.utils.dataset_loader import read_csv_flexible
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", required=True); ap.add_argument("--out_dir", required=True); args = ap.parse_args()
    df = read_csv_flexible(args.dataset, cache_dir="data/cache")
    out = pathlib.Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out/"full.csv", index=False)
    df.head(min(len(df), 100)).to_csv(out/"subset_100.csv", index=False)
    df.head(min(len(df), 20)).to_csv(out/"subset_20.csv", index=False)
    print(f"Wrote subsets to {out}")
if __name__ == "__main__": main()
