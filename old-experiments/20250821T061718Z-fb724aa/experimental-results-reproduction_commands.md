# Reproduction Commands (Actual)

## Environment Setup
```bash
# Working directory
cd /Users/bradleyharaguchi/Algoverse-Self-Correction-Classification
git checkout pivot/teacher-learner-rts

# Python environment (Anaconda)
python --version  # 3.12.7 | packaged by Anaconda, Inc.
pip install -r requirements.txt
pip install -e .
```

## Data Preparation
```bash
# Created math sample dataset
head -21 data/cache/9af2994868fb0e9e_ground_truth_qna.csv > data/math_sample_20.csv
wc -l data/math_sample_20.csv  # 21 lines (20 questions + header)
```

## Experiment 1: Teacher-Learner Pipeline
```bash
export DEMO_MODE=0
export $(cat .env | grep -v '^#' | xargs)
python -m src.main run \
  --dataset "data/math_sample_20.csv" \
  --max-turns 3 \
  --provider openai \
  --out "outputs/experiment_fresh_math20.json"
```

## Experiment 2: GPT-4 Self-Correction Baseline  
```bash
export $(cat .env | grep -v '^#' | xargs)
python scripts/gpt_self_correction_eval.py \
  --error-csv "data/cache/6e7f51856a8b0cd5_error_bias_examples_v3.csv" \
  --qna-csv   "data/math_sample_20.csv" \
  --model "gpt-4o-mini" \
  > outputs/fresh_baseline_math20.jsonl
```

## Validation Run
```bash
export DEMO_MODE=0
export $(cat .env | grep -v '^#' | xargs)
python -m src.main run \
  --dataset "data/math_sample_20.csv" \
  --max-turns 3 \
  --provider openai \
  --out "outputs/test_fixed_math20.json"
```

## Analysis
```bash
# Manual analysis performed with Python scripts
python -c "import json; [analysis code]"
```
