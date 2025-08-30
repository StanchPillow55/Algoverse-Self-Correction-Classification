import pytest, json
from src.evaluation.gsm8k_evaluator import GSM8KEvaluator

def test_gsm8k_extract_ok():
    ev = GSM8KEvaluator()
    text = "We compute step by step. #### 42"
    res = ev.compare(text, "42")
    assert res["em"] == 1.0 and res["diagnosis"] == "correct"

def test_gsm8k_extract_last_number():
    ev = GSM8KEvaluator()
    text = "Reasoning… 10 + 32 = 42. Answer: 43"
    res = ev.compare(text, "42")
    assert res["em"] == 0.0 and res["diagnosis"] in {"arithmetic_slip","logical_flaw","approximation_error"}

def test_gsm8k_no_answer():
    ev = GSM8KEvaluator()
    res = ev.compare("", "5")
    assert res["em"] == 0.0 and res["diagnosis"] == "no_answer"
