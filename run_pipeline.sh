#!/bin/bash
set -euo pipefail

echo "=== 0) Pre-flight: branch + safety around .env ==="
git rev-parse --abbrev-ref HEAD
git fetch --all -p
git checkout pivot/teacher-learner-rts

# Ensure .env is ignored and never committed
if ! grep -qE '(^|/)\.env$' .gitignore 2>/dev/null; then
  echo ".env" >> .gitignore
  echo "Appended .env to .gitignore"
fi
# If .env was ever staged, unstage it now
git rm --cached .env >/dev/null 2>&1 || true

# Load OpenAI key without echoing it
if [ -f .env ]; then
  set +x
  set -a; source .env; set +a
  set -x
  # Do NOT print secrets
else
  echo "Note: .env not found. OpenAI runs will be skipped; demo runs will still execute."
fi

echo "=== 1) Env + deps ==="
python -V
pip -V
if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
mkdir -p outputs

echo "=== 2) Smoke run (DEMO provider) to verify pipeline end-to-end ==="
export DEMO_MODE=1
python -m src.main info || true
python -m src.main run \
  --dataset data/math20.csv \
  --max-turns 2 \
  --out outputs/smoke_demo.json \
  --provider demo

echo "=== 3) Optional: OpenAI-backed run (only if OPENAI_API_KEY is available *and* provider is implemented) ==="
if [ "${OPENAI_API_KEY-}" != "" ]; then
  export DEMO_MODE=0
  # Try an OpenAI run; if the provider is not implemented, continue without failing.
  if python -m src.main run \
      --dataset data/math20.csv \
      --max-turns 3 \
      --out outputs/smoke_openai.json \
      --provider openai; then
    echo "OpenAI run completed."
  else
    echo "OpenAI provider run failed or not implemented. Proceeding with demo results only."
  fi
else
  echo "OPENAI_API_KEY not detected in environment — skipping OpenAI run."
fi

echo "=== 4) Evaluate outputs against the research proposal (no training; trace analytics) ==="
# This analyzer reads any available outputs/*.json and computes:
# - items, final_accuracy_mean
# - mean #turns per item
# - Δaccuracy (final - first) distribution
# - per-bias counts and per-bias average Δaccuracy
# - per-template 'next-turn improvement' rate
python - << 'PY'
import json, glob, statistics as stats, os, sys
from collections import Counter, defaultdict

def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def analyze(traces_obj):
    traces = traces_obj.get("traces", [])
    n = len(traces)
    if n == 0:
        return {"warning": "no traces"}
    # Basic metrics
    finals = [t["final_accuracy"] for t in traces]
    turns_per_item = [len(t["turns"]) for t in traces]
    first_acc = [t["turns"][0]["accuracy"] for t in traces]
    delta_acc = [fin - fst for fin, fst in zip(finals, first_acc)]

    # Bias counts (use last-turn label) and per-bias Δaccuracy
    bias_counts = Counter()
    bias_deltas = defaultdict(list)
    for t in traces:
        last = t["turns"][-1]
        bias = last.get("teacher_bias","None") or "None"
        bias_counts[bias] += 1
        bias_deltas[bias].append(t["final_accuracy"] - t["turns"][0]["accuracy"])

    # Template improvement rate: when a template is used at turn i,
    # did accuracy improve at turn i relative to i-1?
    template_improve = defaultdict(lambda: [0,0])  # [improved, total]
    for t in traces:
        turns = t["turns"]
        for i in range(1, len(turns)):
            tmpl = turns[i].get("template") or "None"
            prev_acc = turns[i-1]["accuracy"]
            cur_acc  = turns[i]["accuracy"]
            if tmpl != "None":
                template_improve[tmpl][1] += 1
                if cur_acc > prev_acc:
                    template_improve[tmpl][0] += 1

    def rate(pair): 
        improved, total = pair
        return 0.0 if total == 0 else improved/total

    summary = {
        "items": n,
        "final_accuracy_mean": sum(finals)/n,
        "turns_mean": stats.mean(turns_per_item),
        "delta_accuracy_mean": sum(delta_acc)/n,
        "improved_items": sum(1 for d in delta_acc if d>0),
        "worsened_items": sum(1 for d in delta_acc if d<0),
        "unchanged_items": sum(1 for d in delta_acc if d==0),
    }

    per_bias = {
        b: {
            "count": bias_counts[b],
            "delta_accuracy_mean": (sum(v)/len(v) if v else 0.0)
        } for b in sorted(bias_counts.keys())
    }

    per_template = {
        t: {
            "uses": pair[1],
            "next_turn_improve_rate": rate(pair)
        } for t, pair in sorted(template_improve.items(), key=lambda kv: kv[0])
    }

    return {"summary": summary, "per_bias": per_bias, "per_template": per_template}

reports = {}
paths = sorted(glob.glob("outputs/*.json"))
if not paths:
    print("No outputs/*.json found; nothing to evaluate.", file=sys.stderr); sys.exit(1)

for p in paths:
    try:
        obj = load(p)
        reports[os.path.basename(p)] = analyze(obj)
    except Exception as e:
        reports[os.path.basename(p)] = {"error": str(e)}

# Write a Markdown report
os.makedirs("outputs", exist_ok=True)
md_path = "outputs/eval_report.md"
with open(md_path, "w", encoding="utf-8") as f:
    f.write("# Teacher/Learner Evaluation Report\n\n")
    for fname, rep in reports.items():
        f.write(f"## {fname}\n\n")
        if "error" in rep:
            f.write(f"Error reading {fname}: {rep['error']}\n\n"); continue
        s = rep["summary"]
        f.write("**Summary**\n\n")
        f.write(f"- items: {s['items']}\n")
        f.write(f"- final_accuracy_mean: {s['final_accuracy_mean']:.3f}\n")
        f.write(f"- mean_turns_per_item: {s['turns_mean']:.2f}\n")
        f.write(f"- delta_accuracy_mean: {s['delta_accuracy_mean']:.3f}\n")
        f.write(f"- improved / worsened / unchanged: {s['improved_items']} / {s['worsened_items']} / {s['unchanged_items']}\n\n")
        f.write("**Per-bias Δaccuracy**\n\n")
        for b, v in rep["per_bias"].items():
            f.write(f"- {b}: count={v['count']}, delta_accuracy_mean={v['delta_accuracy_mean']:.3f}\n")
        f.write("\n**Per-template next-turn improvement rate**\n\n")
        for t, v in rep["per_template"].items():
            f.write(f"- {t}: uses={v['uses']}, next_turn_improve_rate={v['next_turn_improve_rate']:.3f}\n")
        f.write("\n---\n\n")
print(json.dumps(reports, indent=2))
print("\nWrote Markdown report to outputs/eval_report.md")
PY

echo "=== 5) Run smoke tests (functionality first; no cleanup yet) ==="
pytest -q || pytest tests/smoke || true

echo "=== 6) Commit safe changes (NEVER commit .env) ==="
git status --porcelain
# Ensure .env is not staged
if git ls-files --error-unmatch .env >/dev/null 2>&1; then
  echo "ERROR: .env is tracked! Removing from index."; git rm --cached .env
fi

git add -A
git commit -m "run: demo & optional OpenAI pipeline; add evaluation report; ensure .env in .gitignore" || true
git diff --stat HEAD~1

echo "=== NOTE ===
- Do NOT clean legacy code/data yet. Cleanup happens only after we finalize results and Task 9 is handled.
- If Task 9 (config-driven defaults) is still pending, continue using explicit CLI flags as above."

echo "
=== Tip: After you finish ===
1) Run smoke / unit tests:   pytest -q  (or  pytest tests/smoke)
2) Commit:                   git add -A && git commit -m \"<brief, present-tense summary>\"
3) Show diff:                git diff --stat HEAD~1
"
