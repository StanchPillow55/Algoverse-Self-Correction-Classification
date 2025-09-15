# Algoverse — Teacher/Learner RTS Pipeline

This repo provides a teacher–learner pipeline with confidence-aware reprompt selection,
full per-turn trace logging,evaluator bias feedback, and rate-limited OpenAI calls.


## Quickstart

### Environment
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt || pip install -r requirements-dev.txt
```

### .env (API Keys) — required for real runs

Create `.env` (never committed) with:

```
# OpenAI API Key
OPENAI_API_KEY=sk-...    # do not share

# Anthropic API Key
ANTHROPIC_API_KEY=sk-ant-...    # do not share

# HuggingFace API Key (for Llama models)
HUGGINGFACE_API_KEY=hf_...    # do not share

# Replicate API Token (optional)
REPLICATE_API_TOKEN=r8_...    # do not share

# Demo mode for testing without API calls
DEMO_MODE=0
```

### Datasets

The runner accepts local CSV or GitHub URLs. For convenience:

```bash
# Use provided sample (20) if present; else auto-download a full CSV as needed
mkdir -p runs/tmp/inputs
curl -fsSL -o runs/tmp/inputs/full.csv \
  https://raw.githubusercontent.com/StanchPillow55/Algoverse-Self-Correction-Classification/feat/initial-error-table/data/ground_truth_qna.csv
# Create subsets
(head -n 1 runs/tmp/inputs/full.csv && tail -n +2 runs/tmp/inputs/full.csv | head -n 20) > runs/tmp/inputs/subset_20.csv
(head -n 1 runs/tmp/inputs/full.csv && tail -n +2 runs/tmp/inputs/full.csv | head -n 100) > runs/tmp/inputs/subset_100.csv
```


### DEMO_MODE = 1 (no API key required)

```bash
export DEMO_MODE=1 PROVIDER=demo
python -m src.main info
python -m src.main run --dataset runs/tmp/inputs/subset_20.csv  --max-turns 3 --out runs/tmp/subset_20_summary.json --provider "$PROVIDER"
python -m src.main run --dataset runs/tmp/inputs/subset_100.csv --max-turns 3 --out runs/tmp/subset_100_summary.json --provider "$PROVIDER"
# Full (≈1300) if you have a bigger CSV:
python -m src.main run --dataset runs/tmp/inputs/full.csv      --max-turns 3 --out runs/tmp/full_summary.json       --provider "$PROVIDER"
```

### DEMO_MODE = 0 (real API; requires `.env`)

```bash
set -a; source .env; set +a   # will not echo secrets
export DEMO_MODE=0 PROVIDER=openai OPENAI_MODEL=gpt-4o-mini
# Optional rate-limit knobs
export MAX_CONCURRENCY=2 RPS_LIMIT=2 TPM_LIMIT=120000 MAX_RETRIES=6 RETRIES_ENABLED=1

python -m src.main info
python -m src.main run --dataset runs/tmp/inputs/subset_20.csv  --max-turns 3 --out runs/tmp/subset_20_summary.json  --provider "$PROVIDER"
python -m src.main run --dataset runs/tmp/inputs/subset_100.csv --max-turns 3 --out runs/tmp/subset_100_summary.json --provider "$PROVIDER"
python -m src.main run --dataset runs/tmp/inputs/full.csv      --max-turns 3 --out runs/tmp/full_summary.json       --provider "$PROVIDER"
```


## HumanEval Code Generation

This pipeline now supports HumanEval code generation tasks with sandboxed test execution.

### HumanEval DEMO_MODE = 1 (no API key required)

```bash
export DEMO_MODE=1 PROVIDER=demo
python -m src.main info

# Run HumanEval subsets in demo mode
python -m src.main run --dataset humaneval --subset subset_20  --max-turns 3 --out runs/tmp/heval_demo_subset20.json  --provider "$PROVIDER"
python -m src.main run --dataset humaneval --subset subset_100 --max-turns 3 --out runs/tmp/heval_demo_subset100.json --provider "$PROVIDER"
python -m src.main run --dataset humaneval --subset full       --max-turns 3 --out runs/tmp/heval_demo_full.json     --provider "$PROVIDER"
```

### HumanEval DEMO_MODE = 0 (real API; requires `.env`)

```bash
set -a; source .env; set +a   # will not echo secrets
export DEMO_MODE=0 PROVIDER=openai OPENAI_MODEL=gpt-4o-mini

# Run HumanEval with real API
python -m src.main run --dataset humaneval --subset subset_20  --max-turns 3 --out runs/tmp/heval_subset20.json  --provider "$PROVIDER"
python -m src.main run --dataset humaneval --subset subset_100 --max-turns 3 --out runs/tmp/heval_subset100.json --provider "$PROVIDER"
python -m src.main run --dataset humaneval --subset full       --max-turns 3 --out runs/tmp/heval_full.json       --provider "$PROVIDER"
```

### HumanEval Features

* **Sandboxed Execution**: Code is executed in a safe subprocess environment with timeouts and import restrictions
* **Test-Based Scoring**: Solutions are validated by running unit tests, not string matching
* **Pass@1 Metric**: Standard HumanEval evaluation using execution-based correctness
* **Safety Checks**: Code is checked for dangerous patterns before execution
* **Demo Mode**: Simulated execution for testing without actual code running

### HumanEval Subsets

* `subset_20`: First 20 tasks (good for quick testing)
* `subset_100`: First 100 tasks (validation runs)
* `full`: All 164 tasks (complete evaluation)

### Outputs

* Traces: `runs/<RUN_ID>/traces.jsonl` (per-turn prompt, response, bias label, feedback, **test results**)
* Summaries: `runs/<RUN_ID>/*_summary.json`
* Analysis: `runs/<RUN_ID>/analysis.md`, `runs/<RUN_ID>/summary.csv` (if generated)
* HumanEval traces include `execution_details` with test pass/fail information


# New Run Settings
  ## Environment setup
  ```
  export OPENAI_API_KEY="your-key-here"
  export DEMO_MODE=0
  export PROVIDER=openai
  export OPENAI_MODEL=gpt-4o-mini
  ```

  ## Create experiement directories (This is already completed, but you may tweak this if you want new or different output directories)
  ```
  mkdir -p runs/experiments/{baseline,full_system,confidence_only,error_awareness_only,multiturn_only}
  ```

  ## Run with original dataset (or any other dataset)
  ```
  # Run baseline (single turn, no features)
  export RUN_ID=baseline
  python -m src.main run \
    --dataset runs/tmp/inputs/subset_100.csv \
    --max-turns 1 \
    --out runs/experiments/baseline/baseline_summary.json \
    --provider openai \
    --config configs/experiments/baseline.yaml

  # Run full system (all features enabled)
  export RUN_ID=full_system
  python -m src.main run \
    --dataset runs/tmp/inputs/subset_100.csv \
    --max-turns 3 \
    --out runs/experiments/full_system/full_system_summary.json \
    --provider openai \
    --config configs/experiments/full_system.yaml

  # Run confidence only
  export RUN_ID=confidence_only
  python -m src.main run \
    --dataset runs/tmp/inputs/subset_100.csv \
    --max-turns 1 \
    --out runs/experiments/confidence_only/confidence_only_summary.json \
    --provider openai \
    --config configs/experiments/confidence_only.yaml

  # Run error awareness only
  export RUN_ID=error_awareness_only
  python -m src.main run \
    --dataset runs/tmp/inputs/subset_100.csv \
    --max-turns 1 \
    --out runs/experiments/error_awareness_only/error_awareness_only_summary.json \
    --provider openai \
    --config configs/experiments/error_awareness_only.yaml

  # Run multi-turn only
  export RUN_ID=multiturn_only
  python -m src.main run \
    --dataset runs/tmp/inputs/subset_100.csv \
    --max-turns 3 \
    --out runs/experiments/multiturn_only/multiturn_only_summary.json \
    --provider openai \
    --config configs/experiments/multiturn_only.yaml
  ```

  ## (EXAMPLE) Running after incorporating new datasets
  ```
  export RUN_ID=livecodebench_full_system
  python -m src.main run \
    --dataset data/livecodebench_sample.csv \
    --max-turns 3 \
    --out runs/experiments/livecodebench_full_system/summary.json \
    --provider openai \
    --config configs/experiments/full_system.yaml
  # etc.. follow the formatting for RUN_ID and run configs
  ```

### Notes

* `.env` is ignored by git; never commit API keys.
* Analysis for older runs is archived under `/old-experiments/<STAMP>/`.
* HumanEval requires `requests` for dataset downloading (installed via requirements.txt)
* Code execution is sandboxed but still runs locally - exercise caution in production environments


## Latest Results (UTC 2025-08-30T20:43:16.849775Z)

### HumanEval (gpt-4o)
- Results: results/heval_metrics.csv
- Experiments: runs/experiments/*/heval_*.json
- Summary: reports/experiment_summary.md

### GSM8K (gpt-4o, 1000 problems)
- Results: results/gsm8k_metrics.csv  
- Experiments: runs/experiments/*/gsm8k_*.json
- Summary: reports/experiment_summary.md

### Ablation Experiments
- baseline: Single turn, no confidence, plain prompting
- confidence_only: Single turn with confidence scoring
- error_awareness_only: Single turn with error-aware prompting
- multiturn_only: 3 turns with plain prompting
- full_system: 3 turns with confidence and error-aware prompting

### Reproduction
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
set -a; source .env; set +a
export PROVIDER=openai OPENAI_MODEL=gpt-4o MAX_CONCURRENCY=2 RPS_LIMIT=2 TPM_LIMIT=120000 MAX_RETRIES=6 RETRIES_ENABLED=1
python -m src.main run --dataset humaneval --subset subset_20 --max-turns 2 --out runs/smoke/heval20.json --provider "$PROVIDER"
python -m src.main run --dataset runs/tmp/inputs/gsm8k_20.csv --max-turns 2 --out runs/smoke/gsm8k20.json --provider "$PROVIDER"
python -m src.main run --dataset humaneval --subset full --max-turns 3 --out runs/full/heval_full.json --provider "$PROVIDER"
python -m src.main run --dataset runs/tmp/inputs/gsm8k_1k.csv --max-turns 3 --out runs/full/gsm8k_1k.json --provider "$PROVIDER"
```
