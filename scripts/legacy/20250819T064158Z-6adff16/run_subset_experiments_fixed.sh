#!/usr/bin/env bash
#
# SUBSET EXPERIMENT SCRIPT (20 & 100) - FIXED VERSION
# - Runs only subset_20 and subset_100 experiments with full trace logging
# - Properly handles environment variables
#
set -Eeuo pipefail

# Friendly error reporting
trap 'echo "🛑 ERROR: Script failed near line $LINENO. Check logs for details."' ERR

echo "=== SUBSET EXPERIMENTS (20 & 100) - FIXED VERSION ===" 

# Setup environment
cd /Users/bradleyharaguchi/Algoverse-Self-Correction-Classification
source .venv/bin/activate

# Load environment secrets first
if [ -f .env ]; then set -a; . ./.env; set +a; fi

# Setup run environment
GIT_SHA_SHORT=$(git rev-parse --short HEAD)
RUN_ID=$(date -u +"%Y%m%dT%H%M%SZ")-$GIT_SHA_SHORT-fixed-exp
RUN_DIR="runs/$RUN_ID"
mkdir -p "$RUN_DIR/inputs"

echo "🚀 Starting FIXED experiment with RUN_ID: $RUN_ID"

# Prepare dataset subsets
DATASET_URL="data/math_sample_100.csv"
python scripts/make_subsets.py --dataset "$DATASET_URL" --out_dir "$RUN_DIR/inputs"

# Set environment variables EXPLICITLY to override demo mode
export PROVIDER=openai
export OPENAI_MODEL=${OPENAI_MODEL:-gpt-4o-mini}
export DEMO_MODE=0  # Critical: Turn OFF demo mode
export RUN_ID
export GIT_COMMIT="$GIT_SHA_SHORT"

echo "Environment check:"
echo "- PROVIDER: $PROVIDER"
echo "- OPENAI_MODEL: $OPENAI_MODEL" 
echo "- DEMO_MODE: $DEMO_MODE"
echo "- OpenAI API Key set: ${OPENAI_API_KEY:+Yes}"

# Define runner function
run_split() {
  local SPLIT="$1"
  local CSV_PATH="$2"
  echo "--- Running split: $SPLIT ---"
  export DATASET_SPLIT="$SPLIT"
  # Force environment variables in the command
  env DEMO_MODE=0 PROVIDER=openai python -m src.main run \
    --dataset "$CSV_PATH" \
    --max-turns 3 \
    --out "$RUN_DIR/${SPLIT}_summary.json" \
    --provider openai || true
}

# Execute only subset_20 and subset_100
run_split "subset_20" "$RUN_DIR/inputs/subset_20.csv"
run_split "subset_100" "$RUN_DIR/inputs/subset_100.csv"

echo "✅ Experiment runs complete"

# Check if we have traces
if [ -f "$RUN_DIR/traces.jsonl" ]; then
    echo "✅ Found traces file"
    # Generate analysis reports
    echo "=== Generating Analysis ===" 
    python scripts/summarize_traces.py --in_jsonl "$RUN_DIR/traces.jsonl" --out_csv "$RUN_DIR/summary.csv"
    python scripts/diagnostics.py --traces "$RUN_DIR/traces.jsonl" --summary "$RUN_DIR/summary.csv" --out_md "$RUN_DIR/diagnostics.md"
    
    # Validate JSONL
    echo "--- Validating traces.jsonl ---"
    python -c 'import json, sys; [json.loads(line) for line in open(sys.argv[1])]; print("JSONL is valid.")' "$RUN_DIR/traces.jsonl"
    
    echo -e "\n🎉 FIXED subset experiments complete! Results in: $RUN_DIR"
    echo "📄 Traces: $RUN_DIR/traces.jsonl"
    echo "📊 Summary: $RUN_DIR/summary.csv"
    echo "📋 Diagnostics: $RUN_DIR/diagnostics.md"
else
    echo "⚠️  No traces file found - check if logging worked correctly"
    echo "Files in run directory:"
    ls -la "$RUN_DIR/" || true
fi
