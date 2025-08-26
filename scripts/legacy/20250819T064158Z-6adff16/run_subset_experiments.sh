#!/usr/bin/env bash
#
# SUBSET EXPERIMENT SCRIPT (20 & 100)
# - Runs only subset_20 and subset_100 experiments with full trace logging
# - Generates analysis outputs and diagnostics
#
set -Eeuo pipefail

# Friendly error reporting
trap 'echo "🛑 ERROR: Script failed near line $LINENO. Check logs for details."' ERR

echo "=== SUBSET EXPERIMENTS (20 & 100) ===" 

# Setup environment
cd /Users/bradleyharaguchi/Algoverse-Self-Correction-Classification
source .venv/bin/activate

# Ensure trace logger exists
mkdir -p src/utils scripts

# Create TraceLogger if it doesn't exist
if [ ! -f src/utils/trace_logger.py ]; then
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
echo "✅ Created TraceLogger"
fi

# Patch runner if not already patched
python - << 'PY'
from pathlib import Path
p = Path("src/loop/runner.py")
if p.exists():
    t = p.read_text("utf-8")
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
    else:
        print("✅ Runner already patched for trace logging.")
else:
    print("⚠️  src/loop/runner.py not found - will continue anyway")
PY

# Create analysis helper scripts if they don't exist
if [ ! -f scripts/make_subsets.py ]; then
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
fi

if [ ! -f scripts/summarize_traces.py ]; then
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
fi

if [ ! -f scripts/diagnostics.py ]; then
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
fi

echo "✅ Helper scripts created"

# Setup run environment
GIT_SHA_SHORT=$(git rev-parse --short HEAD)
RUN_ID=$(date -u +"%Y%m%dT%H%M%SZ")-$GIT_SHA_SHORT-subset-exp
RUN_DIR="runs/$RUN_ID"
mkdir -p "$RUN_DIR/inputs"

echo "🚀 Starting experiment with RUN_ID: $RUN_ID"

# Prepare dataset subsets
DATASET_URL="data/math_sample_100.csv"
python scripts/make_subsets.py --dataset "$DATASET_URL" --out_dir "$RUN_DIR/inputs"

# Load environment secrets
if [ -f .env ]; then set -a; . ./.env; set +a; fi

# Set environment variables
export PROVIDER=openai
export OPENAI_MODEL=${OPENAI_MODEL:-gpt-4o-mini}
export DEMO_MODE=0
export RUN_ID
export GIT_COMMIT="$GIT_SHA_SHORT"

# Define runner function
run_split() {
  local SPLIT="$1"
  local CSV_PATH="$2"
  echo "--- Running split: $SPLIT ---"
  export DATASET_SPLIT="$SPLIT"
  python -m src.main run --dataset "$CSV_PATH" --max-turns 3 --out "$RUN_DIR/${SPLIT}_summary.json" --provider openai || true
}

# Execute only subset_20 and subset_100
run_split "subset_20" "$RUN_DIR/inputs/subset_20.csv"
run_split "subset_100" "$RUN_DIR/inputs/subset_100.csv"

echo "✅ Experiment runs complete"

# Generate analysis reports
echo "=== Generating Analysis ===" 
python scripts/summarize_traces.py --in_jsonl "$RUN_DIR/traces.jsonl" --out_csv "$RUN_DIR/summary.csv"
python scripts/diagnostics.py --traces "$RUN_DIR/traces.jsonl" --summary "$RUN_DIR/summary.csv" --out_md "$RUN_DIR/diagnostics.md"

# Validate JSONL
echo "--- Validating traces.jsonl ---"
python -c 'import json, sys; [json.loads(line) for line in open(sys.argv[1])]; print("JSONL is valid.")' "$RUN_DIR/traces.jsonl"

echo -e "\n🎉 Subset experiments complete! Results in: $RUN_DIR"
echo "📄 Traces: $RUN_DIR/traces.jsonl"
echo "📊 Summary: $RUN_DIR/summary.csv"
echo "📋 Diagnostics: $RUN_DIR/diagnostics.md"
