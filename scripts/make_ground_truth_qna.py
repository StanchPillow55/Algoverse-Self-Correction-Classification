import argparse, csv, random, re
from pathlib import Path

from datasets import load_dataset  # pip install datasets

def parse_gsm8k_final(ans: str) -> str:
    # GSM8K answers end with "#### <final>"
    m = re.search(r"####\s*(.*)$", ans.strip(), flags=re.S)
    return (m.group(1).strip() if m else ans.strip())

def sample_split(ds, split: str, n: int, seed: int):
    data = ds[split]
    n = min(n, len(data))
    idxs = list(range(len(data)))
    random.Random(seed).shuffle(idxs)
    return [data[i] for i in idxs[:n]]

def fetch_gsm8k(per_split: int, seed: int):
    # GSM8K default config is "main"
    ds = load_dataset("openai/gsm8k", "main")
    rows = []
    for split in ("train", "test"):
        if split in ds:
            for ex in sample_split(ds, split, per_split, seed):
                q = ex["question"].strip()
                a = parse_gsm8k_final(ex["answer"])
                rows.append((q, a))
    return rows

def fetch_hotpotqa(per_split: int, seed: int):
    # Use the common "distractor" setting
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor")
    rows = []
    for split in ("validation", "train"):  # dev first, then train
        if split in ds:
            for ex in sample_split(ds, split, per_split, seed):
                q = ex["question"].strip()
                a = str(ex["answer"]).strip()
                rows.append((q, a))
    return rows

def fetch_boolq(per_split: int, seed: int):
    ds = load_dataset("google/boolq")
    rows = []
    for split in ("validation", "train"):
        if split in ds:
            for ex in sample_split(ds, split, per_split, seed):
                q = ex["question"].strip()
                a = "Yes" if bool(ex["answer"]) else "No"  # map bool to Yes/No
                rows.append((q, a))
    return rows

def fetch_humaneval(per_split: int, seed: int):
    # HumanEval only has 'test'
    ds = load_dataset("openai/openai_humaneval")
    rows = []
    split = "test"
    if split in ds:
        for ex in sample_split(ds, split, per_split, seed):
            q = ex["prompt"].strip()            # treat prompt as Question
            a = ex["canonical_solution"].strip()  # code as Answer
            rows.append((q, a))
    return rows

def main():
    ap = argparse.ArgumentParser("Build Question,Answer ground-truth CSV from public datasets.")
    ap.add_argument("--out", default="data/ground_truth_qna.csv")
    ap.add_argument("--per", type=int, default=200, help="Rows per split per dataset (capped by split size)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    all_rows = []
    all_rows += fetch_gsm8k(args.per, args.seed)
    all_rows += fetch_hotpotqa(args.per, args.seed)
    all_rows += fetch_boolq(args.per, args.seed)
    all_rows += fetch_humaneval(args.per, args.seed)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Question", "Answer"])
        w.writerows(all_rows)

    print(f"Wrote {len(all_rows)} rows to {args.out}")

if __name__ == "__main__":
    main()
