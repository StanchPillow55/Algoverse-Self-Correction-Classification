#!/usr/bin/env python3
import argparse, json, pandas as pd
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--in_jsonl", required=True); ap.add_argument("--out_csv", required=True); args = ap.parse_args()
    rows = []
    for line in open(args.in_jsonl, "r"):
        ex = json.loads(line)
        turns = ex.get("turns", [])
        confs = [t.get("model_reported_confidence") for t in turns if t.get("model_reported_confidence") is not None]
        rows.append({"problem_id": ex["problem_id"], "split": ex["dataset_split"], "final_correct": ex["final_correct"],
                     "num_turns": ex["num_turns"], "first_try_correct": bool(turns and turns[0].get("is_correct")),
                     "recoveries": bool(not (turns and turns[0].get("is_correct")) and ex["final_correct"]),
                     "avg_confidence": sum(confs)/len(confs) if confs else None})
    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print(f"Wrote summary to {args.out_csv}")
if __name__ == "__main__": main()
