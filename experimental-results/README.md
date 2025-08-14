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

### Inputs provided for this report

* `experiment_fresh_math20.json` — Teacher–Learner run (Final Acc = 0.30).
* `teacher_learner_validation.json` — Validation Teacher–Learner run (Final Acc = 0.30).
* `fresh_baseline_math20.jsonl` — GPT-4 baseline run (Final Acc = 0.15).
* `experiment_metadata.json` — Env, parameters, datasets, commit info.
* `reproduction_commands.md` — Exact shell used to produce the runs.
* `caveats.md` — Known deviations and limitations.
