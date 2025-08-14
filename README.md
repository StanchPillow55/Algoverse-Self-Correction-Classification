# Confidence-Aware Reprompt Selection for Reliable LLM Self-Correction
*(Teacher-Bot / Learner-Bot Edition)*

This repository implements a classification-free **Teacher–Learner** architecture for reliable LLM
self-correction. A **Learner-Bot** attempts an answer; a **Teacher-Bot** (a second LLM) detects broad
failure modes (**Anchoring, Confirmation, Availability/Bandwagon, Hindsight, Overgeneralization**),
estimates confidence, and uses an **RTS (Reprompt Template Selector)** to decide *whether* and *how*
to reprompt — all **without training** new ML models.

The pipeline runs on small ground-truth QA CSVs (≥20 rows), logs rich traces, and supports URL-based
datasets hosted on GitHub (auto-converts `blob/<branch>/...` → `raw.githubusercontent.com/...`).
This documentation reflects the code in the branch `pivot/teacher-learner-rts`.

---

## Features

- **Teacher–Learner loop** with confidence- & bias-aware reprompt selection
- **RTS prompt library** (`rts_templates.json`) with style & cognitive-load metadata
- **STOP rules** (max turns, no improvement, high-confidence correct, negative expected Δ)
- **URL-aware dataset loader** (local files or GitHub URLs) with caching
- **Two evaluators**
  - End-to-end Teacher–Learner runner (`python -m src.main run`)
  - GPT-4 self-correction baseline (`scripts/gpt_self_correction_eval.py`)
- **Trace-centric logging** (per turn accuracy, bias labels, tokens, chosen template)
- **Smoke/unit tests** and runnable **demo** (provider=`demo`) for quick verification

---

## Repository Layout

```text
├── configs/                        # Experiment configs (datasets/exp grids)
├── data/                           # Local datasets (e.g., math20.csv)
├── experimental-results/           # Reproducible analysis of current results
│   ├── raw/                        # Drop raw logs here (JSON/JSONL)
│   ├── processed/                  # Auto-generated metrics tables
│   ├── figures/                    # Auto-generated plots
│   ├── analysis.md                 # Narrative report (updated by analyzer)
│   ├── README.md                   # How to refresh results
│   ├── experiment_metadata.json    # Provided metadata
│   ├── reproduction_commands.md    # Provided exact commands
│   └── caveats.md                  # Known caveats/deviations
├── legacy/                         # Quarantined legacy code (not used by pipeline)
├── scripts/
│   ├── analyze_results.py          # Aggregates logs → CSV + plots + report
│   └── gpt_self_correction_eval.py # Baseline: GPT-4 self-correction JSONL
├── src/
│   ├── agents/                     # teacher.py, learner.py
│   ├── loop/                       # runner.py (orchestration, STOP rules)
│   └── utils/                      # dataset_loader.py (GitHub URL→raw, cache)
├── tests/                          # Smoke/unit tests
├── rts_templates.json              # Prompt library for RTS
├── requirements.txt                # Python dependencies
├── Makefile                        # Convenience targets (env, eval, analyze)
└── README.md
```

---

## TL;DR Quickstart (minimal demo)

```bash
# 1) Clone
git clone https://github.com/StanchPillow55/Algoverse-Self-Correction-Classification.git
cd Algoverse-Self-Correction-Classification
git checkout pivot/teacher-learner-rts

# 2) Environment (lightweight)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# 3) Dry run (DEMO provider, no API required)
export DEMO_MODE=1
python -m src.main info
python -m src.main run \
  --dataset data/math20.csv \
  --max-turns 2 \
  --out outputs/demo_run.json \
  --provider demo

# 4) Smoke tests
pytest -q || pytest tests/smoke
```

---

## Environment Setup

Choose one.

### Option A — Conda + pip (recommended)

```bash
conda create -n algoverse-trl python=3.12 -y
conda activate algoverse-trl
pip install -r requirements.txt
pip install -e .
python -c "import sys,platform;print(sys.version);print(platform.platform())"
```

### Option B — uv (fast, reproducible lock)

```bash
# Requires uv: https://github.com/astral-sh/uv
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
pip install -e .
```

**Security**: Ensure `.env` is in `.gitignore` (it is). **Never commit** `.env` or tokens.

---

## Data

### CSV Schema (auto-mapped, case-insensitive)

* **Required:** `question`, `reference`
* **Optional:** `qid`, `topic`

Example (`data/math20.csv`):

```csv
qid,question,reference
q1,What is 12 + 7?,19
q2,If a box has 3 apples and you add 5, how many?,8
...
```

### GitHub-Hosted CSVs (use blob URLs)

The loader supports GitHub `blob/<branch>/...` links and converts to **raw** automatically.

```bash
# Example: ground-truth QnA hosted on a feature branch
export QNA_URL="https://github.com/StanchPillow55/Algoverse-Self-Correction-Classification/blob/feat/initial-error-table/data/ground_truth_qna.csv"

# Run the pipeline directly on the URL:
python -m src.main run \
  --dataset "$QNA_URL" \
  --max-turns 3 \
  --out outputs/url_run.json \
  --provider demo
```

---

## Configuration

* **RTS templates:** `rts_templates.json` — list of prompts with metadata
  `{ "id", "text", "style", "cognitive_load", "length", "bias" }`
* **Experiment grids:** `configs/experiments/*.yaml`
* **STOP rules & loop:** set via CLI flags + internal defaults

**Common CLI flags**

```bash
python -m src.main run \
  --dataset <path-or-github-url> \
  --max-turns 2 \
  --out outputs/run.json \
  --provider demo|openai
```

For OpenAI runs: place `OPENAI_API_KEY=...` in `.env`, then `export DEMO_MODE=0`.

---

## End-to-End Pipeline

### Orchestration

1. Load dataset (local or GitHub URL).
2. For each item: Learner produces `answer` + `self_conf`.
3. Teacher labels `teacher_bias` + `teacher_conf`; RTS selects reprompt or STOP.
4. Iterate up to `T` turns or until a STOP rule triggers.
5. Log turns and compute accuracy vs. reference.

```mermaid
flowchart TD
  A[CSV (local or GitHub URL)] --> B[Loader (auto-map headers, cache)]
  B --> C[Runner (per item loop)]
  C --> D[Learner-Bot (k samples, self_conf)]
  C --> E[Teacher-Bot (bias, teacher_conf)]
  E --> F[RTS selection (template, stop?)]
  F -->|reprompt| D
  F -->|stop| G[Finalize answer]
  G --> H[Trace Logger (turns, Δaccuracy, tokens)]
  H --> I[Analysis scripts / reports]
```

### Commands

#### Teacher–Learner (evaluation; no training)

```bash
# Demo (no API)
export DEMO_MODE=1
python -m src.main run \
  --dataset data/math20.csv \
  --max-turns 2 \
  --out outputs/demo_teacher_learner.json \
  --provider demo
```

```bash
# OpenAI (real LLMs)
export DEMO_MODE=0    # ensure we do not use the demo path
# Ensure .env contains OPENAI_API_KEY (do not commit .env)
python -m src.main run \
  --dataset "data/math_sample_20.csv" \
  --max-turns 3 \
  --out "outputs/experiment_fresh_math20.json" \
  --provider openai
```

#### GPT-4 Self-Correction Baseline

```bash
python scripts/gpt_self_correction_eval.py \
  --error-csv "data/cache/6e7f51856a8b0cd5_error_bias_examples_v3.csv" \
  --qna-csv   "data/math_sample_20.csv" \
  --model "gpt-4o-mini" \
  > outputs/fresh_baseline_math20.jsonl
```

> The exact reproduction shell used for your reported runs is included under
> `experimental-results/reproduction_commands.md`.

---

## Logging & Checkpoints

* **Teacher–Learner traces (JSON):** `outputs/*.json`
  Per item: `{"qid","question","reference","turns":[{"answer","self_conf","teacher_bias","teacher_conf","template","accuracy"}], "final_accuracy", "tokens_total"}`
* **Self-correction baseline (JSONL):** `outputs/*.jsonl`
  One JSON object per item, then a final `{"__summary__": ...}`.
* **Debug helpers:** `outputs/openai_debug.json` (sanitized), `outputs/mismatches.log`

No model checkpoints are produced (classification-free approach).

---

## Reproducibility

* Determinism bounded by the LLM provider & sampling.
* Recommended settings:

  * Learner (math): `temperature=0.2–0.4`
  * Teacher: `temperature=0.0–0.2`
  * Max turns `T ∈ {2,3}`
* Pin environment and record commit hashes. Your provided run used:

  * Python 3.12.7 (Anaconda), macOS ARM64, OpenAI v1.97.1, pandas 2.2.3.

---

## Experimental Results (current)

**Dataset:** Math-20 sample (20 word problems).
**Models:** GPT-4o-mini for both teacher and learner.
**Parameters:** `max_turns=3`, temperatures `0.2/0.0`, `max_tokens=40`.

**Teacher–Learner (OpenAI):** Final accuracy **30%** over 20 items (`experiment_fresh_math20.json`).
**Validation run:** Independent run confirms **30%** over 20 items (`teacher_learner_validation.json`).
**GPT-4 baseline (self-correction):** Final accuracy **15%** over 20 items (`fresh_baseline_math20.jsonl`).

| Method                      | Items | Final Acc |
| --------------------------- | ----: | --------: |
| Teacher–Learner (OpenAI)    |    20 |      0.30 |
| GPT-4 Baseline (1-shot fix) |    20 |      0.15 |

**Net improvement:** **+15 percentage points** (2× over baseline).
For caveats (sample size, bias labeling, template utility, determinism), see `experimental-results/caveats.md`.

To regenerate the aggregate tables/plots, see **Experimental Results** below.

---

## Troubleshooting & FAQs

**CSV on GitHub 404s**
Use a **blob** link; the loader converts to **raw** and caches. (Implemented in `src/utils/dataset_loader.py`.)

**Answers are always "0" in OpenAI mode**
Ensure `export DEMO_MODE=0` and `--provider openai`. Check `outputs/openai_debug.json` and `outputs/mismatches.log`.

**Add a new RTS prompt**
Edit `rts_templates.json` and rerun. Keep metadata (`style`, `cognitive_load`, `length`, `bias`) consistent.

**Secrets**
Only in `.env` (ignored by Git). Never commit secrets.

---

## Experimental Results (reproducible analysis)

A curated, reproducible analysis lives under **`experimental-results/`**.

```bash
# Place raw logs here (copy/move your outputs):
#   experimental-results/raw/experiment_fresh_math20.json
#   experimental-results/raw/fresh_baseline_math20.jsonl
#   experimental-results/raw/teacher_learner_validation.json

# Generate tables, plots, and update the narrative:
python scripts/analyze_results.py \
  --raw experimental-results/raw \
  --out experimental-results
```

This creates:

* `experimental-results/processed/metrics_summary.csv`
* `experimental-results/figures/*.png`
* Updates `experimental-results/analysis.md` with a metrics table

---

## Quality Gates & Self-Checks

* ✅ Commands run without undefined paths
* ✅ All file paths exist or are created by scripts
* ✅ `experimental-results/processed/metrics_summary.csv` produced by analyzer
* ✅ `experimental-results/figures/*.png` produced (if data available)
* ✅ Seeds and versions documented (see metadata)
* ✅ No placeholders remain except those marked `<!-- TODO -->`
* ✅ **Security:** `.env` is `.gitignore`'d; examples never echo secrets

---

## Citation

```
@software{algoverse_teacher_learner_rts_2025,
  title  = {Confidence-Aware Reprompt Selection for Reliable LLM Self-Correction (Teacher–Learner–RTS)},
  author = {Algoverse Team},
  year   = {2025},
  url    = {https://github.com/StanchPillow55/Algoverse-Self-Correction-Classification}
}
```

## License

MIT (see `LICENSE`).
