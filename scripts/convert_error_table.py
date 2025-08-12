#!/usr/bin/env python3
"""
convert_error_table.py

Converts the 4-column human-labeled table:
  Prompt (question), Feedback (LLM Answer), Error (human labelled), Accuracy (0 or 1)

Outputs:
  --out-validation (default: data/validation_dataset.csv)
      columns: sample_id,prompt,response,error_type,accuracy
  --out-initial-gt (default: data/labels/initial_gt.csv)
      columns: dataset,qid,input_prompt,initial_answer,initial_correct,error_label
  --out-dist (default: data/labels/error_distribution.jsonl)
      JSONL lines: {"sample_id": "...", "error_dist": {"overthinking": 0.6, "perfectionism": 0.4}}

Mixture format in Error cell:
  - single:  "overthinking"
  - mixture: "overthinking:0.6; perfectionism:0.4"
Numbers are optional; if absent, a single label gets prob=1.0. Mixtures are normalized if they don't sum to 1.

Usage:
  python scripts/convert_error_table.py \
    --src data/labels/initial_table.csv \
    --out-validation data/validation_dataset.csv \
    --out-initial-gt data/labels/initial_gt.csv \
    --out-dist data/labels/error_distribution.jsonl
"""
import argparse, json, sys
import pandas as pd
from pathlib import Path

ALLOWED = {"no_error","underthinking","overthinking","perfectionism","logic_computation","factuality"}

def parse_error_cell(cell: str):
    if not isinstance(cell, str) or not cell.strip():
        return {"no_error": 1.0}
    s = cell.strip()
    if ";" not in s and ":" not in s:
        lab = s.lower().strip()
        return {lab: 1.0}
    parts = [p.strip() for p in s.split(";") if p.strip()]
    dist = {}
    for p in parts:
        if ":" in p:
            lab, val = p.split(":", 1)
            lab = lab.strip().lower()
            try:
                w = float(val.strip())
            except ValueError:
                w = 0.0
        else:
            lab = p.strip().lower()
            w = 1.0
        dist[lab] = dist.get(lab, 0.0) + w
    total = sum(dist.values()) or 1.0
    dist = {k: v/total for k, v in dist.items()}
    return dist

def dominant_label(dist: dict) -> str:
    if not dist:
        return "no_error"
    return max(dist.items(), key=lambda kv: kv[1])[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Path to 4-column CSV")
    ap.add_argument("--out-validation", default="data/validation_dataset.csv")
    ap.add_argument("--out-initial-gt", default="data/labels/initial_gt.csv")
    ap.add_argument("--out-dist", default="data/labels/error_distribution.jsonl")
    args = ap.parse_args()

    df = pd.read_csv(args.src)
    expected = ["Prompt (question)", "Feedback (LLM Answer)", "Error (human labelled)", "Accuracy (0 or 1)"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        sys.exit(f"Missing expected columns: {missing}\nFound: {list(df.columns)}")

    # parse distributions + dominant label
    dists = [parse_error_cell(x) for x in df["Error (human labelled)"].tolist()]
    doms  = [dominant_label(d) for d in dists]

    # validation_dataset.csv (repo-friendly)
    val = pd.DataFrame({
        "sample_id": [f"row_{i:05d}" for i in range(len(df))],
        "prompt": df["Prompt (question)"],
        "response": df["Feedback (LLM Answer)"],
        "error_type": doms,
        "accuracy": df["Accuracy (0 or 1)"].astype(int),
    })
    Path(args.out_validation).parent.mkdir(parents=True, exist_ok=True)
    val.to_csv(args.out_validation, index=False)

    # initial_gt.csv (for detector training)
    gt = pd.DataFrame({
        "dataset": "manual_v0",
        "qid": val["sample_id"],
        "input_prompt": df["Prompt (question)"],
        "initial_answer": df["Feedback (LLM Answer)"],
        "initial_correct": df["Accuracy (0 or 1)"].astype(int),
        "error_label": doms,
    })
    Path(args.out_initial_gt).parent.mkdir(parents=True, exist_ok=True)
    gt.to_csv(args.out_initial_gt, index=False)

    # error_distribution.jsonl
    Path(args.out_dist).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_dist, "w", encoding="utf-8") as f:
        for sid, dist in zip(val["sample_id"], dists):
            filtered = {k: float(v) for k, v in dist.items() if k in ALLOWED}
            ssum = sum(filtered.values()) or 1.0
            filtered = {k: v/ssum for k, v in filtered.items()}
            f.write(json.dumps({"sample_id": sid, "error_dist": filtered}, ensure_ascii=False) + "\n")

    print(f"Wrote: {args.out_validation}")
    print(f"Wrote: {args.out_initial_gt}")
    print(f"Wrote: {args.out_dist}")

if __name__ == "__main__":
    main()
