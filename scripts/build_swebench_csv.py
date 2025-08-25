#!/usr/bin/env python3
# scripts/build_swebench_csv.py
"""
Build a SWE-bench CSV with columns: Question,Answer

Default pulls from 'princeton-nlp/SWE-bench' (has problem_statement + patch).
You can switch to the Verified/Lite variants with --dataset-name, but note that
some variants (e.g., princeton-nlp/SWE-bench_Verified) may omit the patch field.
"""

import argparse
import csv
from pathlib import Path

import pandas as pd
from datasets import load_dataset

def write_csv(df: pd.DataFrame, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Question", "Answer"])
        for _, row in df.iterrows():
            w.writerow([row["Question"], row["Answer"]])
    print(f"[OK] Wrote {out_path} ({len(df)} rows)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-name", default="princeton-nlp/SWE-bench",
                    help="HF dataset name (e.g., 'princeton-nlp/SWE-bench', 'SWE-bench/SWE-bench_Verified')")
    ap.add_argument("--split", default="test",
                    help="split name (some variants have only a single split; try 'test' or leave empty)")
    ap.add_argument("--out", default="data/swebench.csv", help="output CSV path")
    ap.add_argument("--max-rows", type=int, default=0, help="cap number of rows (0 = no cap)")
    ap.add_argument("--truncate-answer", type=int, default=0,
                    help="truncate the patch to N characters to keep the CSV small (0 = no truncation)")
    args = ap.parse_args()

    # Load
    if args.split:
        ds = load_dataset(args.dataset_name, split=args.split)
    else:
        ds = load_dataset(args.dataset_name)

    df = ds.to_pandas()

    # Choose fields
    if "problem_statement" not in df.columns:
        raise SystemExit("Dataset is missing 'problem_statement' (issue text).")
    question = df["problem_statement"].astype(str)

    # Prefer full code patch if present; otherwise fall back to test_patch or empty
    if "patch" in df.columns:
        answer = df["patch"].astype(str)
    elif "test_patch" in df.columns:
        answer = df["test_patch"].astype(str)
    else:
        # Some Verified variants only ship problem_statement + base_commit.
        answer = pd.Series([""] * len(df), dtype=str)

    if args.max_rows:
        question = question.head(args.max_rows)
        answer = answer.head(args.max_rows)

    if args.truncate_answer and args.truncate_answer > 0:
        answer = answer.str.slice(0, args.truncate_answer)

    out_df = pd.DataFrame({"Question": question, "Answer": answer})
    write_csv(out_df, Path(args.out))

if __name__ == "__main__":
    main()
