#!/usr/bin/env python3
import argparse, json, pandas as pd, collections
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--traces", required=True); ap.add_argument("--summary", required=True); ap.add_argument("--out_md", required=True); args = ap.parse_args()
    df = pd.read_csv(args.summary)
    rec_rate = df["recoveries"].mean()
    acc_by_split = df.groupby("split")["final_correct"].mean().to_dict()
    
    with open(args.out_md, "w") as f:
        f.write("# Diagnostics Report\n\n")
        f.write(f"## Recovery Rate (Wrong -> Correct)\n- {rec_rate:.3f}\n\n")
        f.write("## Final Accuracy by Split\n")
        for split, acc in acc_by_split.items():
            f.write(f"- {split}: {acc:.3f}\n")
    print(f"Wrote diagnostics to {args.out_md}")
if __name__ == "__main__": main()
