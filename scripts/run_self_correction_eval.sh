#!/usr/bin/env bash
set -Eeuo pipefail
mkdir -p outputs
python scripts/gpt_self_correction_eval.py \
  --error-csv "https://github.com/StanchPillow55/Algoverse-Self-Correction-Classification/blob/main/data/error_bias_examples_v3.csv" \
  --qna-csv   "https://github.com/StanchPillow55/Algoverse-Self-Correction-Classification/blob/main/data/ground_truth_qna.csv" \
  > outputs/self_correction.jsonl
echo "Wrote outputs/self_correction.jsonl"
