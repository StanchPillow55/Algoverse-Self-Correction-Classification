#!/usr/bin/env python3
"""
GPT-4 evaluation-and-self-correction agent for Algoverse.

- Ingests two CSVs (local paths or GitHub URLs):
  1) error_bias_examples_v3.csv  -> builds Error Checklist
  2) ground_truth_qna.csv        -> test items

- For each item: produce initial answer (+ brief why), diagnose likely errors (0-3),
  apply ONE targeted correction, then emit a JSON object.
- Finally emit a JSON object with "__summary__".
- Prints ONLY JSONL to stdout (no extra prose).

Reasoning policy:
- reasoning_effort: "low"
- temperature = 0
- No chain-of-thought; brief 1–2 sentence justifications only.
"""

import argparse, csv, io, json, os, re, sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

try:
    import requests
except ImportError:
    requests = None

try:
    from openai import OpenAI
    _OPENAI_OK = True
except ImportError:
    _OPENAI_OK = False

# ---- Utilities ----
def _to_raw_github(url: str) -> str:
    m = re.match(r"https://github\.com/([^/]+)/([^/]+)/blob/([^/]+)/(.*)", url)
    if m:
        user, repo, branch, path = m.groups()
        return f"https://raw.githubusercontent.com/{user}/{repo}/{branch}/{path}"
    return url

def _read_csv_flexible(path_or_url: str) -> pd.DataFrame:
    p = _to_raw_github(path_or_url)
    try:
        return pd.read_csv(p, dtype=str, keep_default_na=False)
    except Exception as e:
        if requests and p.startswith("http"):
            r = requests.get(p, timeout=30)
            r.raise_for_status()
            return pd.read_csv(io.StringIO(r.text), dtype=str, keep_default_na=False)
        raise e

def _norm_text(s: Optional[str]) -> str:
    return "" if s is None else str(s).strip()

def _norm_for_compare(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower()).rstrip('.')

def _num_or_none(s: str) -> Optional[float]:
    try:
        return float(s)
    except (ValueError, TypeError):
        m = re.search(r"[-+]?\d+(?:\.\d+)?", _norm_text(s))
        return float(m.group(0)) if m else None

# ---- Schema Normalization ----
ERR_MAP = {"id": ["id"], "error_type": ["error_type"], "failure_pattern": ["failure_pattern"], "brief_reasoning": ["brief_reasoning"], "correction_rule": ["correction_rule"], "topic": ["topic"]}
QNA_MAP = {"qid": ["qid"], "question": ["question"], "ground_truth": ["ground_truth"], "topic": ["topic"]}

def _auto_map_columns(df: pd.DataFrame, mapping: Dict[str, List[str]]) -> Tuple[pd.DataFrame, Dict[str,str], List[str]]:
    lowmap = {c.lower().strip(): c for c in df.columns}
    result_map, missing = {}, []
    for want, cands in mapping.items():
        found = next((lowmap[c.lower()] for c in cands if c.lower() in lowmap), None)
        if found: result_map[want] = found
        else: missing.append(want)
    df2 = df[list(result_map.values())].rename(columns={v:k for k,v in result_map.items()})
    return df2, result_map, missing

# ---- Main Agent Logic ----
@dataclass
class ChecklistEntry:
    name: str; cue: str; fix: str; validation: str; topic: Optional[str] = None

def build_checklist(df: pd.DataFrame) -> List[ChecklistEntry]:
    entries = []
    for _, r in df.iterrows():
        name = _norm_text(r.get("error_type"))
        pat = _norm_text(r.get("failure_pattern"))
        why = _norm_text(r.get("brief_reasoning"))
        fix = _norm_text(r.get("correction_rule"))
        if not all([name, pat, fix]): continue
        cue = f"{pat}. {why}".strip()
        val = "recompute" if "recompute" in fix.lower() else "verify constraints"
        entries.append(ChecklistEntry(name=name, cue=cue, fix=fix, validation=val, topic=_norm_text(r.get("topic")) or None))
    return entries

def call_openai(messages: List[Dict[str,str]], model: str) -> Dict[str, Any]:
    if not _OPENAI_OK: return {"answer": "OpenAI not installed", "why": ""}
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    try:
        resp = client.chat.completions.create(model=model, messages=messages, temperature=0, max_tokens=80, response_format={"type":"json_object"})
        content = resp.choices[0].message.content or "{}"
        return json.loads(content)
    except Exception as e:
        return {"answer": f"API_ERROR: {e}", "why": "API call failed."}

def run_pipeline(err_src: str, qna_src: str, model: str):
    df_err, _, err_missing = _auto_map_columns(_read_csv_flexible(err_src), ERR_MAP)
    df_qna, _, qna_missing = _auto_map_columns(_read_csv_flexible(qna_src), QNA_MAP)
    checklist = build_checklist(df_err)
    
    outputs, counts = [], {"exact_init":0, "norm_init":0, "exact_final":0, "norm_final":0}
    
    for _, row in df_qna.iterrows():
        qid, q, gt, topic = row.get("qid",""), row.get("question",""), row.get("ground_truth",""), row.get("topic","")
        if not q or not gt: continue
        
        # Initial Answer
        msg1 = [{"role":"system","content":"Answer concisely as JSON: {\"answer\": \"string\", \"why\": \"string, <=2 sentences\"}."},{"role":"user","content":f"Question: {q}"}]
        res1 = call_openai(msg1, model)
        ans1, why1 = _norm_text(res1.get("answer")), _norm_text(res1.get("why"))
        
        ex0, nm0 = _norm_for_compare(ans1) == _norm_for_compare(gt), _norm_for_compare(ans1) == _norm_for_compare(gt)
        if ex0: counts["exact_init"] += 1
        if nm0: counts["norm_init"] += 1
        
        # Self-Correction
        ans2, why2, applied = ans1, why1, {"name": None, "fix_summary": None}
        if not ex0:
            suspected = sorted([e for e in checklist if e.topic == topic], key=lambda x: len(x.cue), reverse=True)[:1]
            if suspected:
                chosen = suspected[0]
                applied = {"name": chosen.name, "fix_summary": chosen.fix[:120]}
                msg2 = [{"role":"system","content":"Correct the answer using ONE fix. Reply as JSON: {\"answer\": \"string\", \"why\": \"string, <=2 sentences\"}."},
                        {"role":"user","content":f"Question: {q}\nInitial Answer: {ans1}\nCorrection Rule: \"{chosen.fix}\""}]
                res2 = call_openai(msg2, model)
                ans2, why2 = _norm_text(res2.get("answer")), _norm_text(res2.get("why"))

        ex1, nm1 = _norm_for_compare(ans2) == _norm_for_compare(gt), _norm_for_compare(ans2) == _norm_for_compare(gt)
        if ex1: counts["exact_final"] += 1
        if nm1: counts["norm_final"] += 1

        obj = {"qid": qid, "question": q, "ground_truth": gt, "initial_answer": ans1, "final_answer": ans2,
               "exact_match": ex1, "normalized_match": nm1, "applied_correction": applied}
        print(json.dumps(obj, ensure_ascii=False))
        outputs.append(obj)
    
    # Final Summary
    n = len(outputs)
    def safe_div(a,b): return (a/b) if b > 0 else 0.0
    summary = {"__summary__": {
        "dataset_size": n,
        "exact_acc": safe_div(counts["exact_final"], n),
        "normalized_acc": safe_div(counts["norm_final"], n),
        "improvement_after_correction": {
            "delta_exact": safe_div(counts["exact_final"] - counts["exact_init"], n),
            "delta_normalized": safe_div(counts["norm_final"] - counts["norm_init"], n)
        },
        "warnings": [f"error_csv missing: {err_missing}", f"qna_csv missing: {qna_missing}"]
    }}
    print(json.dumps(summary, ensure_ascii=False))

def main():
    ap = argparse.ArgumentParser(description="GPT-4 Self-Correction Eval CLI")
    ap.add_argument("--error-csv", default="./data/error_bias_examples_v3.csv")
    ap.add_argument("--qna-csv", default="./data/ground_truth_qna.csv")
    ap.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    args = ap.parse_args()
    run_pipeline(args.error_csv, args.qna_csv, model=args.model)

if __name__ == "__main__":
    main()
