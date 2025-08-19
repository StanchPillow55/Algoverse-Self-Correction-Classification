#!/usr/bin/env python3
"""
Token usage estimation script that scans traces.jsonl files from experiments.
Produces per-run and per-split CSV files plus Markdown report.
"""
import os, sys, json, glob, math, csv, re
from datetime import datetime

OUT_DIR = "experimental-results/token_estimates"
os.makedirs(OUT_DIR, exist_ok=True)

# Try tiktoken; fallback to approx tokens ~= ceil(len(text)/4)
def _get_tokenizer():
    try:
        import tiktoken
        # Use cl100k_base as a reasonable proxy for GPT-4 class models
        return tiktoken.get_encoding("cl100k_base"), "tiktoken-cl100k"
    except Exception:
        return None, "heuristic-4chars"

tok, tok_name = _get_tokenizer()

def count_tokens(txt:str)->int:
    if not txt: return 0
    if tok:
        try:
            return len(tok.encode(txt))
        except Exception:
            pass
    # fallback: conservative heuristic
    return math.ceil(len(txt)/4)

def infer_split_from_path(p:str)->str:
    m = re.search(r"(subset_20|subset_100|full)", p)
    return m.group(1) if m else "unknown"

def infer_split_from_json(obj)->str:
    sp = obj.get("dataset_split") or obj.get("split") or ""
    if isinstance(sp,str) and sp.strip(): return sp.strip()
    return "unknown"

def maybe_infer_full_by_csv(run_root:str)->str:
    # Check runs/<id>/inputs/*.csv for counts
    inp_dir = os.path.join(run_root, "inputs")
    if not os.path.isdir(inp_dir): return "unknown"
    for cand in ["full.csv","subset_100.csv","subset_20.csv"]:
        fp = os.path.join(inp_dir, cand)
        if os.path.isfile(fp):
            try:
                # crude line count
                n = sum(1 for _ in open(fp, encoding="utf-8", errors="ignore"))
                # subtract header if present; still fine as heuristic
                n_eff = max(0, n-1)
                if "subset_20" in cand and n_eff<=25: return "subset_20"
                if "subset_100" in cand and 60<=n_eff<=200: return "subset_100"
                if "full" in cand and n_eff>=1000: return "full"
            except Exception:
                pass
    return "unknown"

def run_id_from_path(p:str)->str:
    # Expect runs/<RUN_ID>/traces.jsonl or .../old-experiments/<STAMP>/<RUN>/traces.jsonl
    parts = p.split(os.sep)
    if "runs" in parts:
        i = parts.index("runs")
        if i+1 < len(parts): return parts[i+1]
    # fallback: last dir name
    return os.path.basename(os.path.dirname(p))

def gather_trace_files():
    pats = [
        "runs/*/traces.jsonl",
        "experimental-results/old-experiments/*/*/traces.jsonl",
        "experimental-results/old-experiments/*/traces.jsonl"
    ]
    files = []
    for pat in pats:
        files.extend(glob.glob(pat))
    return sorted(set(files))

def main():
    rows_per_run = []
    agg_by_split = {}

    files = gather_trace_files()
    if not files:
        print("No traces.jsonl found under runs/ or experimental-results/old-experiments/", file=sys.stderr)
        sys.exit(1)

    for path in files:
        run_id = run_id_from_path(path)
        run_root = os.path.dirname(path)
        split_hint = infer_split_from_path(path)
        n_examples = 0
        in_tokens = 0
        out_tokens = 0
        turns_total = 0
        splits_seen = set()

        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if not line.strip(): continue
                try:
                    obj = json.loads(line)
                except Exception:
                    # tolerate and skip malformed lines
                    continue
                n_examples += 1
                sp = infer_split_from_json(obj)
                if sp == "unknown":
                    sp = split_hint
                if sp == "unknown":
                    sp = maybe_infer_full_by_csv(run_root)
                splits_seen.add(sp)

                turns = obj.get("turns") or []
                for t in turns:
                    turns_total += 1
                    prompt = t.get("prompt","")
                    resp   = t.get("response_text","") or t.get("answer","")
                    in_tokens  += count_tokens(prompt)
                    out_tokens += count_tokens(resp)

        # Pick one split label: prefer explicit; else hint; else inferred
        if len(splits_seen - {"unknown"}) == 1:
            split_final = list(splits_seen - {"unknown"})[0]
        elif split_hint != "unknown":
            split_final = split_hint
        else:
            split_final = maybe_infer_full_by_csv(run_root)

        rows_per_run.append({
            "run_id": run_id,
            "path": path,
            "tokenizer": tok_name,
            "dataset_split": split_final or "unknown",
            "examples": n_examples,
            "turns": turns_total,
            "input_tokens": in_tokens,
            "output_tokens": out_tokens,
            "total_tokens": in_tokens + out_tokens,
        })

    # Aggregate per split
    for r in rows_per_run:
        sp = r["dataset_split"]
        d = agg_by_split.setdefault(sp, {"runs":0,"examples":0,"turns":0,"input_tokens":0,"output_tokens":0,"total_tokens":0})
        d["runs"] += 1
        d["examples"] += r["examples"]
        d["turns"] += r["turns"]
        d["input_tokens"] += r["input_tokens"]
        d["output_tokens"] += r["output_tokens"]
        d["total_tokens"] += r["total_tokens"]

    # Write per_run.csv
    per_run_csv = os.path.join(OUT_DIR, "per_run.csv")
    with open(per_run_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["run_id","dataset_split","examples","turns","input_tokens","output_tokens","total_tokens","tokenizer","path"])
        for r in rows_per_run:
            w.writerow([r["run_id"], r["dataset_split"], r["examples"], r["turns"], r["input_tokens"], r["output_tokens"], r["total_tokens"], r["tokenizer"], r["path"]])

    # Write per_split.csv
    per_split_csv = os.path.join(OUT_DIR, "per_split.csv")
    with open(per_split_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset_split","runs","examples","turns","input_tokens","output_tokens","total_tokens","tokenizer"])
        for sp, d in sorted(agg_by_split.items()):
            w.writerow([sp, d["runs"], d["examples"], d["turns"], d["input_tokens"], d["output_tokens"], d["total_tokens"], tok_name])

    # Markdown report
    report_md = os.path.join(OUT_DIR, "report.md")
    lines = []
    lines.append(f"# Token Usage Report\n")
    lines.append(f"- Generated: {datetime.utcnow().isoformat()}Z")
    lines.append(f"- Tokenizer: {tok_name}")
    lines.append("")
    lines.append("## Per-split totals\n")
    lines.append("| split | runs | examples | turns | input_tokens | output_tokens | total_tokens |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for sp, d in sorted(agg_by_split.items()):
        lines.append(f"| {sp} | {d['runs']} | {d['examples']} | {d['turns']} | {d['input_tokens']} | {d['output_tokens']} | {d['total_tokens']} |")
    lines.append("\n## Per-run details\n")
    lines.append("| run_id | split | examples | turns | input_tokens | output_tokens | total_tokens |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for r in rows_per_run:
        lines.append(f"| {r['run_id']} | {r['dataset_split']} | {r['examples']} | {r['turns']} | {r['input_tokens']} | {r['output_tokens']} | {r['total_tokens']} |")
    open(report_md,"w",encoding="utf-8").write("\n".join(lines))

    print("Wrote:", per_run_csv)
    print("Wrote:", per_split_csv)
    print("Wrote:", report_md)

if __name__ == "__main__":
    main()
