import os
import json
from src.utils.normalizer_gsm8k import normalize_numeric_string, gsm8k_em

def test_normalize_commas_and_whitespace():
    assert normalize_numeric_string(" 1,234 ") == "1234"
    assert normalize_numeric_string("1,234.00") == "1234"


def test_normalize_signs_and_decimals():
    assert normalize_numeric_string("+3.50") == "3.5"
    assert normalize_numeric_string("-3/2").startswith("-")


def test_units_and_symbols():
    assert normalize_numeric_string("$1,000.00") == "1000"
    assert normalize_numeric_string("50%") == "50"


def test_em_various_formats_match():
    gold = "The answer is #### 1,234.00\n"
    preds = ["1234", " 1,234 ", "#### 1,234.0"]
    for p in preds:
        assert gsm8k_em(p, gold) == 1


def test_em_mismatch():
    assert gsm8k_em("123", "#### 124") == 0

