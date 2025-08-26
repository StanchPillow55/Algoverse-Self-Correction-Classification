#!/usr/bin/env bash
#
# SAFE & ROBUST PIPELINE RUNNER (v3)
# - Securely loads .env without echo
# - Runs demo mode, then a guarded OpenAI run
# - Checks for 0% accuracy failure in OpenAI run
# - Uses a separate, robust Python script for analysis
# - Runs only non-disruptive smoke tests
#
set -Eeuo pipefail

# --- Configuration ---
BRANCH="pivot/teacher-learner-rts"
DATASET="data/math20.csv"
ANALYZER_SCRIPT="scripts/analyze_outputs.py"

# --- Error Handling ---
# Provides a helpful error message if any command fails
trap 'echo "🛑 ERROR: Pipeline failed near line $LINENO. Check logs above for details."' ERR

# --- Pipeline Steps ---
echo "=== 1) Setup: Branch and .env Safety ==="
git checkout "$BRANCH"
touch .gitignore
grep -qE '(^|/)\.env$' .gitignore || echo ".env" >> .gitignore
git rm --cached .env >/dev/null 2>&1 || true # Unstage if ever tracked

echo "=== 2) Dependencies and Environment ==="
# Load .env quietly and export variables for child processes
if [ -f .env ]; then
    set -a; . ./.env; set +a
    echo "✅ Loaded .env variables."
fi
[ -f requirements.txt ] && pip install -q -r requirements.txt
mkdir -p outputs scripts

echo "=== 3) Core Pipeline: Demo Run ==="
export DEMO_MODE=1
python -m src.main run --dataset "$DATASET" --max-turns 2 --out "outputs/smoke_demo.json" --provider demo

echo "=== 4) Core Pipeline: Optional OpenAI Run (Guarded) ==="
if [ -n "${OPENAI_API_KEY-}" ]; then
    export DEMO_MODE=0
    echo "🔑 OPENAI_API_KEY detected. Attempting OpenAI run..."
    if python -m src.main run --dataset "$DATASET" --max-turns 3 --out "outputs/smoke_openai.json" --provider openai; then
        # Sanity-check the output accuracy to detect provider failures
        acc=$(python -c 'import json; f="outputs/smoke_openai.json"; o=json.load(open(f)); print(o.get("final_accuracy_mean", ""))')
        if [[ "$acc" == "0.0" || "$acc" == "0" ]]; then
            echo "⚠️ WARNING: OpenAI run produced 0% accuracy. The provider is likely failing. Investigate and check API key/credits."
        else
            echo "✅ OpenAI run completed with final accuracy: $acc"
        fi
    else
        echo "⚠️ OpenAI provider run failed to execute. Proceeding with demo results only."
    fi
else
    echo "⚪ OPENAI_API_KEY not set. Skipping OpenAI run."
fi

echo "=== 5) Analysis and Reporting ==="
if [ ! -f "$ANALYZER_SCRIPT" ]; then
    echo "🔎 Analyzer script not found. Please create $ANALYZER_SCRIPT."
    exit 1
fi
python "$ANALYZER_SCRIPT" outputs/*.json

echo "=== 6) Testing: Smoke Tests Only ==="
pytest tests/smoke -q

echo "=== 7) Finalizing: Commit Safe Changes ==="
# Ensure .env is never staged or committed
git ls-files --error-unmatch .env >/dev/null 2>&1 && git rm --cached .env || true
# Add the runner, analyzer, and results
git add -A run_pipeline.sh scripts/analyze_outputs.py outputs/ .gitignore
git commit -m "feat(pipeline): run safe v3 pipeline and commit results" || echo "No new changes to commit."
git diff --stat HEAD~1

echo ""
echo "🎉 Pipeline finished successfully. Review the report at outputs/eval_report.md"