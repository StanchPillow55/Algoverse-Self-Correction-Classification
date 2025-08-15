# Experimental Results

This directory contains a **reproducible analysis** of recent experiments.

## How to refresh

1. **Drop raw logs** into `experimental-results/raw/`
   - Teacher–Learner traces: JSON files (e.g., `experiment_fresh_math20.json`)
   - Self-correction baseline: JSONL (e.g., `fresh_baseline_math20.jsonl`)
   - Optional validation runs: JSON (e.g., `teacher_learner_validation.json`)

2. **Run analysis**
```bash
python scripts/analyze_results.py \
  --raw experimental-results/raw \
  --out experimental-results
```

3. **Outputs**

* `processed/metrics_summary.csv` — aggregated metrics across runs
* `figures/*.png` — plots (final accuracy, mean turns)
* `analysis.md` — narrative with auto-appended tables

## 🚨 Critical Research Finding

**Confidence-aware reprompt selection DOES NOT SCALE.** While small samples showed promise (2× improvement), full-scale evaluation revealed catastrophic failure (42% worse than baseline).

### Complete Results Summary
| Dataset | N | Teacher-Learner | GPT-4 Baseline | Ratio | Status |
|---------|---|-----------------|----------------|-------|--------|
| Math-20 | 20 | 30.0% | 15.0% | **2.10×** | ✅ Promising |
| Math-100 | 100 | 32.0% | 25.0% | **1.28×** | ⚠️ Weakening |
| **Full** | **1364** | **17.6%** | **30.4%** | **0.58×** | ❌ **Failed** |

**Key Insight**: Small-scale validation can be dangerously misleading. This research demonstrates the critical importance of full-scale evaluation before drawing conclusions.

### Raw Data Files

#### Small-Scale Results (Historical)
* `experiment_fresh_math20.json` — Teacher–Learner run (Final Acc = 0.30)
* `teacher_learner_validation.json` — Validation run (Final Acc = 0.30)
* `fresh_baseline_math20.jsonl` — GPT-4 baseline (Final Acc = 0.15)
* `teacher_learner_math100.json` — Math-100 teacher-learner (Final Acc = 0.32)
* `baseline_math100.jsonl` — Math-100 baseline (Final Acc = 0.25)

#### **Full-Scale Results (Definitive)**
* `teacher_learner_full1364.json` — **Complete dataset teacher-learner (Final Acc = 0.176)**
* `baseline_full1364.jsonl` — **Complete dataset baseline (Final Acc = 0.304)**

#### Documentation
* `experiment_metadata.json` — Environment, parameters, datasets, commit info
* `reproduction_commands.md` — Exact commands used for all experiments
* `caveats.md` — Known limitations and deviations
* `FINAL_SUMMARY.md` — **Complete analysis and research conclusions**
