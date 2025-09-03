# GSM8K Numeric Extraction Fix

This repository addresses prior numeric extraction issues that could yield near-0 accuracy by:

- Parsing answers after the last `####` token when available
- Falling back to the last numeric token in the text
- Normalizing commas, signs, decimals, mixed fractions, and percent/monetary symbols

Implementation: see `src/metrics/accuracy.py` functions:
- `extract_final_answer`
- `gsm8k_extract_gold_answer`
- `gsm8k_em`
- `normalize_numeric_string`

Unit tests: `tests/test_gsm8k_normalizer.py` cover comma removal, signs, decimals, units, and EM behavior.

