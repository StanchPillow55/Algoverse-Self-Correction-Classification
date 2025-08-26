#!/usr/bin/env python3
"""
Aggregate Algoverse Teacher–Learner and GPT self-correction outputs.

Inputs:
  --raw <dir>   : directory containing raw logs (JSON, JSONL)

Outputs (under --out):
  processed/metrics_summary.csv
  figures/*.png
  analysis.md (auto-appends a metrics table if missing)
"""
import argparse, json, os, pathlib, statistics as stats
from typing import Any, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _is_jsonl(p: pathlib.Path) -> bool:
    return p.suffix.lower() in (".jsonl", ".ndjson")

def _load_json(p: pathlib.Path) -> Any:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def _iter_jsonl(p: pathlib.Path):
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue

def summarize_teacher_learner(obj: Dict[str, Any]) -> Dict[str, Any]:
    traces = obj.get("traces") or []
    if not traces and "items" in obj and "final_accuracy_mean" in obj:
        return {
            "items": obj.get("items"),
            "initial_acc": None,
            "final_acc": obj.get("final_accuracy_mean"),
            "delta_acc": None,
            "mean_turns": None,
            "type": "teacher_learner",
        }
    n = len(traces)
    if n == 0:
        return {"items": 0, "type": "teacher_learner"}
    finals = [t.get("final_accuracy", 0) for t in traces]
    firsts = [t.get("turns", [{"accuracy":0}])[0].get("accuracy", 0) for t in traces]
    turns = [len(t.get("turns", [])) for t in traces]
    tokens = [t.get("tokens_total", 0) or 0 for t in traces]
    return {
        "items": n,
        "initial_acc": sum(firsts)/n if n else None,
        "final_acc": sum(finals)/n if n else None,
        "delta_acc": (sum(finals)/n - sum(firsts)/n) if n else None,
        "mean_turns": stats.mean(turns) if turns else None,
        "acc_per_1k_tokens": ((sum(finals)/n) / ((sum(tokens)/n)/1000.0)) if sum(tokens)>0 else None,
        "type": "teacher_learner",
    }

def summarize_self_correction(p: pathlib.Path) -> Dict[str, Any]:
    n = 0
    final = 0.0
    for obj in _iter_jsonl(p):
        if "__summary__" in obj:
            continue
        exact = bool(obj.get("exact_match", False))
        norm = bool(obj.get("normalized_match", False))
        if exact or norm:
            final += 1.0
        n += 1
    if n == 0:
        return {"items": 0, "type": "self_correction"}
    return {
        "items": n,
        "initial_acc": None,
        "final_acc": final / n,
        "delta_acc": None,
        "mean_turns": 1.0,
        "acc_per_1k_tokens": None,
        "type": "self_correction",
    }

def analyze_dir(raw_dir: pathlib.Path) -> pd.DataFrame:
    rows = []
    for p in sorted(raw_dir.glob("*")):
        if p.is_dir():
            continue
        try:
            if p.suffix.lower() == ".json":
                s = summarize_teacher_learner(_load_json(p))
                s["run"] = p.name
                rows.append(s)
            elif _is_jsonl(p):
                s = summarize_self_correction(p)
                s["run"] = p.name
                rows.append(s)
        except Exception as e:
            rows.append({"run": p.name, "error": str(e)})
    return pd.DataFrame(rows)

def save_plots(df: pd.DataFrame, out_dir: pathlib.Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    # Final accuracy by run
    try:
        sub = df.dropna(subset=["final_acc"])
        if not sub.empty:
            fig = plt.figure()
            xs = np.arange(len(sub))
            plt.bar(xs, sub["final_acc"])
            plt.xticks(xs, sub["run"], rotation=45, ha="right")
            plt.ylabel("Final Accuracy")
            plt.title("Final Accuracy by Run")
            fig.tight_layout()
            fig.savefig(out_dir / "final_accuracy_by_run.png", dpi=150)
            plt.close(fig)
    except Exception:
        pass
    # Mean turns by run
    try:
        sub = df.dropna(subset=["mean_turns"])
        if not sub.empty:
            fig = plt.figure()
            xs = np.arange(len(sub))
            plt.bar(xs, sub["mean_turns"])
            plt.xticks(xs, sub["run"], rotation=45, ha="right")
            plt.ylabel("Mean Turns")
            plt.title("Mean Turns by Run")
            fig.tight_layout()
            fig.savefig(out_dir / "mean_turns_by_run.png", dpi=150)
            plt.close(fig)
    except Exception:
        pass

def append_analysis_md(df: pd.DataFrame, analysis_md: pathlib.Path):
    lines = []
    lines.append("\n## Auto-generated Metrics\n\n")
    if df.empty:
        lines.append("_No recognizable runs found in raw logs._\n")
    else:
        cols = ["run","type","items","initial_acc","final_acc","delta_acc","mean_turns","acc_per_1k_tokens","error"]
        for c in cols:
            if c not in df.columns:
                df[c] = ""
        df2 = df[cols].copy().fillna("")
        lines.append("| " + " | ".join(cols) + " |\n")
        lines.append("|" + "|".join(["---"]*len(cols)) + "|\n")
        for _, r in df2.iterrows():
            vals = [str(r[c]) for c in cols]
            lines.append("| " + " | ".join(vals) + " |\n")
    if analysis_md.exists():
        existing = analysis_md.read_text(encoding="utf-8")
        if "## Auto-generated Metrics" in existing:
            start = existing.find("## Auto-generated Metrics")
            existing = existing[:start]
            analysis_md.write_text(existing + "".join(lines), encoding="utf-8")
            return
    analysis_md.write_text("".join(lines), encoding="utf-8")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    raw_dir = pathlib.Path(args.raw)
    out_root = pathlib.Path(args.out)
    proc = out_root / "processed"
    figs = out_root / "figures"
    analysis_md = out_root / "analysis.md"

    out_root.mkdir(parents=True, exist_ok=True)
    proc.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    df = analyze_dir(raw_dir)
    if not df.empty:
        df.to_csv(proc / "metrics_summary.csv", index=False)
    save_plots(df, figs)
    append_analysis_md(df, analysis_md)
    print("Wrote:", proc / "metrics_summary.csv", "and figures to", figs)

if __name__ == "__main__":
    main()
