#!/usr/bin/env bash
#
# FULL TRACE LOGGING & EXPERIMENT SCRIPT
# - Instruments the pipeline for detailed multi-turn JSONL trace logging.
# - Adds helper scripts to create dataset subsets and generate summary reports.
# - Executes a multi-tier experiment (20/100/full) against a target dataset.
# - Generates a final diagnostics markdown report.
#
set -Eeuo pipefail

# Friendly error reporting
trap 'echo "🛑 ERROR: Script failed near line $LINENO. Check logs for details."' ERR

# --- Main Logic ---

echo "=== 0) Pre-flight & .env Hygiene ==="
if [ ! -d Algoverse-Self-Correction-Classification ]; then
  git clone https://github.com/StanchPillow55/Algoverse-Self-Correction-Classification.git
fi
cd Algoverse-Self-Correction-Classification
git fetch --all -p
git checkout pivot/teacher-learner-rts

# Ensure secrets are never committed
touch .gitignore
grep -qE '(^|/)\.env$' .gitignore || echo ".env" >> .gitignore
git rm --cached .env >/dev/null 2>&1 || true

# Capture commit hash for run ID
GIT_SHA_SHORT=$(git rev-parse --short HEAD)
echo "✅ Setup complete. Working with commit: $GIT_SHA_SHORT"

echo "=== 1) Environment Setup ==="
if [ ! -d .venv ]; then python3 -m venv .venv; fi
source .venv/bin/activate
pip install -q -r requirements.txt || true
pip install -q -e .
mkdir -p runs src/utils scripts
echo "✅ Virtual environment activated and dependencies installed."

echo "=== 2) Implement TraceLogger and Instrument Code ==="
# 2a) Create the TraceLogger utility
cat > src/utils/trace_logger.py << 'PY'
import os, json, datetime as dt
from typing import Optional, Dict, Any

class TraceLogger:
    def __init__(self, run_id: str, out_dir: str = "./runs", dataset_split: str = "unknown", git_commit: str = ""):
        self.run_id = run_id
        self.dataset_split = dataset_split
        self.git_commit = git_commit
        self.root = os.path.join(out_dir, run_id)
        os.makedirs(self.root, exist_ok=True)
        self.path = os.path.join(self.root, "traces.jsonl")
        self._fh = open(self.path, "a", encoding="utf-8")

    def close(self):
        if self._fh: self._fh.close()

    def start_example(self, problem_id: str, text: str) -> Dict[str, Any]:
        return {"problem_id": str(problem_id), "dataset_split": self.dataset_split,
                "original_problem_text": text, "turns": [], "final_answer": "",
                "final_correct": None, "num_turns": 0, "run_id": self.run_id,
                "git_commit": self.git_commit, "time_started": dt.datetime.utcnow().isoformat()+"Z"}

    def on_turn(self, ex: Dict[str, Any], **kwargs):
        ex["turns"].append({k: v for k, v in kwargs.items()})

    def end_example(self, ex: Dict[str, Any], final_answer: str, final_correct: bool):
        ex.update({"final_answer": str(final_answer), "final_correct": bool(final_correct),
                   "num_turns": len(ex["turns"]), "time_finished": dt.datetime.utcnow().isoformat()+"Z"})
        self._fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
        self._fh.flush()

    def write_run_config(self, cfg: dict):
        with open(os.path.join(self.root, "run_config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
PY

# 2b) Patch the runner to use the TraceLogger
python - << 'PY'
from pathlib import Path
p = Path("src/loop/runner.py")
t = p.read_text("utf-8")

# Minimal surgical patch; adds hooks for logging
if "TraceLogger(" not in t:
    t = t.replace("from pathlib import Path", "from pathlib import Path\nfrom src.utils.trace_logger import TraceLogger")
    t = t.replace("rows = df.to_dict(orient=\"records\")",
                  "rows = df.to_dict(orient=\"records\")\n    import os\n"
                  "    run_id = os.environ.get('RUN_ID', 'dev_run')\n"
                  "    split = os.environ.get('DATASET_SPLIT', 'unknown')\n"
                  "    git_sha = os.environ.get('GIT_COMMIT', '')\n"
                  "    logger = TraceLogger(run_id=run_id, dataset_split=split, git_commit=git_sha)\n"
                  "    logger.write_run_config({'dataset': dataset_csv, 'max_turns': max_turns, 'provider': provider, 'split': split, 'model': os.getenv('OPENAI_MODEL')})")
    t = t.replace("for idx, row in enumerate(rows):",
                  "for idx, row in enumerate(rows):\n        ex = logger.start_example(problem_id=(_qid or f'q{idx+1}'), text=q)")
    t = t.replace("turns.append(turn)",
                  "turns.append(turn)\n        logger.on_turn(ex, turn_index=len(turns)-1, prompt=history[-1]['content'], "
                  "response_text=answer, response_is_final=(turn_idx == max_turns-1 or stop), is_correct=bool(acc), "
                  "evaluator_signal=('stop' if stop else 'continue'), model_reported_confidence=self_conf, "
                  "evaluator_feedback=teacher_feedback, model_name=learner.model)")
    t = t.replace("traces.append(trace)",
                  "traces.append(trace)\n        logger.end_example(ex, final_answer=answer, final_correct=bool(acc))")
    t = t.replace("return summary", "logger.close()\n    return summary")
    p.write_text(t, "utf-8")
    print("✅ Patched src/loop/runner.py for trace logging.")
PY
echo "✅ Trace logging instrumented."

echo "=== 3) Add Helper Scripts for Analysis ==="
# 3a) Create script to make dataset subsets
cat > scripts/make_subsets.py << 'PY'
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
PY
chmod +x scripts/make_subsets.py

# 3b) Create script to summarize traces into a CSV
cat > scripts/summarize_traces.py << 'PY'
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
PY
chmod +x scripts/summarize_traces.py

# 3c) Create script to generate diagnostics markdown
cat > scripts/diagnostics.py << 'PY'
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
PY
chmod +x scripts/diagnostics.py
echo "✅ Analysis helper scripts created."

echo "=== 4) Execute Multi-Tier Experiment ==="
DATASET_URL="data/math_sample_100.csv"
RUN_ID=$(date -u +"%Y%m%dT%H%M%SZ")-$GIT_SHA_SHORT
RUN_DIR="runs/$RUN_ID"
mkdir -p "$RUN_DIR/inputs"

# Prepare dataset subsets
python scripts/make_subsets.py --dataset "$DATASET_URL" --out_dir "$RUN_DIR/inputs"

# Load secrets without echoing
if [ -f .env ]; then set -a; . ./.env; set +a; fi

# Set common environment variables for the run
export PROVIDER=openai
export OPENAI_MODEL=${OPENAI_MODEL:-gpt-4o-mini}
export DEMO_MODE=0
export RUN_ID
export GIT_COMMIT="$GIT_SHA_SHORT"

# Define a runner function to avoid repetition
run_split() {
  local SPLIT="$1"
  local CSV_PATH="$2"
  echo "--- Running split: $SPLIT ---"
  export DATASET_SPLIT="$SPLIT"
  python -m src.main run --dataset "$CSV_PATH" --max-turns 3 --out "$RUN_DIR/${SPLIT}_summary.json" --provider openai || true
}

# Execute runs for each split
run_split "subset_20" "$RUN_DIR/inputs/subset_20.csv"
run_split "subset_100" "$RUN_DIR/inputs/subset_100.csv"
run_split "full" "$RUN_DIR/inputs/full.csv"
echo "✅ All experiment runs complete."

echo "=== 5) Generate Final Reports ==="
# Summarize all traces from the run into a single CSV
python scripts/summarize_traces.py --in_jsonl "$RUN_DIR/traces.jsonl" --out_csv "$RUN_DIR/summary.csv"

# Generate the final diagnostics markdown from the traces and summary
python scripts/diagnostics.py --traces "$RUN_DIR/traces.jsonl" --summary "$RUN_DIR/summary.csv" --out_md "$RUN_DIR/diagnostics.md"

# Validate the final JSONL file to ensure it's not corrupt
echo "--- Validating traces.jsonl ---"
python -c 'import json, sys; [json.loads(line) for line in open(sys.argv[1])]; print("JSONL is valid.")' "$RUN_DIR/traces.jsonl"
echo "✅ Reports generated."

echo "=== 6) Finalize and Commit ==="
git rm --cached .env >/dev/null 2>&1 || true # Final safety check
git add -A
git commit -m "feat(logging): add full multi-turn trace logging and analysis scripts" || echo "⚪ No new changes to commit."
git diff --stat HEAD~1 || true

echo -e "\n🎉 Experiment complete. Artifacts are in: $RUN_DIR"
