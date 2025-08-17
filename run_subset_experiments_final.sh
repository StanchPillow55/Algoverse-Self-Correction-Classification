#!/usr/bin/env bash
#
# FINAL SUBSET EXPERIMENT SCRIPT (20 & 100)
# - Fixed demo mode issue in main.py
# - Runs only subset_20 and subset_100 experiments with full trace logging
#
set -Eeuo pipefail

# Friendly error reporting
trap 'echo "🛑 ERROR: Script failed near line $LINENO. Check logs for details."' ERR

echo "=== FINAL SUBSET EXPERIMENTS (20 & 100) ===" 

# Setup environment
cd /Users/bradleyharaguchi/Algoverse-Self-Correction-Classification
source .venv/bin/activate

# Load environment secrets first
if [ -f .env ]; then set -a; . ./.env; set +a; fi

# Setup run environment
GIT_SHA_SHORT=$(git rev-parse --short HEAD)
RUN_ID=$(date -u +"%Y%m%dT%H%M%SZ")-$GIT_SHA_SHORT-final-exp
RUN_DIR="runs/$RUN_ID"
mkdir -p "$RUN_DIR/inputs"

echo "🚀 Starting FINAL experiment with RUN_ID: $RUN_ID"

# Prepare dataset subsets
DATASET_URL="data/math_sample_100.csv"
python scripts/make_subsets.py --dataset "$DATASET_URL" --out_dir "$RUN_DIR/inputs"

# Set environment variables EXPLICITLY
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

# Test a single question first to verify everything works
echo "=== Quick Test ===" 
python -c "
import os
from src.agents.learner import LearnerBot
os.environ['DEMO_MODE'] = '0'
os.environ['PROVIDER'] = 'openai'
learner = LearnerBot('openai')
result = learner.answer('What is 2 + 3?', [], None)
print(f'Test result: {result}')
if result[0] != '0':
    print('✅ OpenAI API is working!')
else:
    print('❌ Still getting demo mode response')
    exit(1)
"

# Define runner function
run_split() {
  local SPLIT="$1"
  local CSV_PATH="$2"
  echo "--- Running split: $SPLIT ---"
  export DATASET_SPLIT="$SPLIT"
  python -m src.main run \
    --dataset "$CSV_PATH" \
    --max-turns 3 \
    --out "$RUN_DIR/${SPLIT}_summary.json" \
    --provider openai || true
}

# Execute only subset_20 and subset_100
run_split "subset_20" "$RUN_DIR/inputs/subset_20.csv"
run_split "subset_100" "$RUN_DIR/inputs/subset_100.csv"

echo "✅ Experiment runs complete"

# Check if we have traces and generate analysis
if [ -f "$RUN_DIR/traces.jsonl" ]; then
    echo "✅ Found traces file"
    
    # Check if we have actual real responses (not all "0")
    REAL_RESPONSES=$(head -5 "$RUN_DIR/traces.jsonl" | grep -o '"response_text": "[^"]*"' | grep -v '"response_text": "0"' | wc -l)
    echo "Real responses found: $REAL_RESPONSES"
    
    if [ "$REAL_RESPONSES" -gt 0 ]; then
        echo "✅ Found real OpenAI responses!"
    else
        echo "⚠️  Still getting demo responses - check configuration"
    fi
    
    # Generate analysis reports
    echo "=== Generating Analysis ===" 
    python scripts/summarize_traces.py --in_jsonl "$RUN_DIR/traces.jsonl" --out_csv "$RUN_DIR/summary.csv"
    python scripts/diagnostics.py --traces "$RUN_DIR/traces.jsonl" --summary "$RUN_DIR/summary.csv" --out_md "$RUN_DIR/diagnostics.md"
    
    # Validate JSONL
    echo "--- Validating traces.jsonl ---"
    python -c 'import json, sys; [json.loads(line) for line in open(sys.argv[1])]; print("JSONL is valid.")' "$RUN_DIR/traces.jsonl"
    
    echo -e "\n🎉 FINAL subset experiments complete! Results in: $RUN_DIR"
    echo "📄 Traces: $RUN_DIR/traces.jsonl"
    echo "📊 Summary: $RUN_DIR/summary.csv"
    echo "📋 Diagnostics: $RUN_DIR/diagnostics.md"
    
    # Show a sample of results
    echo -e "\nSample results:"
    head -2 "$RUN_DIR/summary.csv"
else
    echo "⚠️  No traces file found - check if logging worked correctly"
    echo "Files in run directory:"
    ls -la "$RUN_DIR/" || true
fi
