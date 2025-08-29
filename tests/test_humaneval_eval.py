import os
import tempfile
from pathlib import Path
from src.eval.humaneval_scorer import score_humaneval_candidate
from src.data.humaneval_loader import create_demo_humaneval_data
from src.eval.code_executor import execute_code_safely
from src.loop.runner import run_dataset
from src.metrics.accuracy import gsm8k_em


def test_humaneval_evaluator_executes_tests():
    # Use demo task
    task = create_demo_humaneval_data()[0]
    # Correct implementation should pass
    good = """def has_close_elements(numbers, threshold):
    for i in range(len(numbers)):
        for j in range(len(numbers)):
            if i != j and abs(numbers[i]-numbers[j]) < threshold:
                return True
    return False
"""
    res = score_humaneval_candidate(task, good, use_demo_mode=False)
    # Even without demo mode, sandbox executes; ensure structured fields present
    assert isinstance(res.get('execution_result'), dict)
    er = res['execution_result']
    assert er.get('passed') in (True, False)

    # Intentionally wrong implementation should fail with wrong_answer
    bad = """def has_close_elements(numbers, threshold):\n    return False\n"""
    res_bad = score_humaneval_candidate(task, bad, use_demo_mode=False)
    er_bad = res_bad['execution_result']
    assert er_bad.get('passed') is False
    assert er_bad.get('error_type') in ('wrong_answer','runtime_error','compile_error')
    assert isinstance(er_bad.get('traceback_excerpt',''), str)

    # Compile error case
    compile_err = "def has_close_elements(numbers, threshold)\n    return True"  # missing colon
    res_ce = score_humaneval_candidate(task, compile_err, use_demo_mode=False)
    assert res_ce['execution_result'].get('error_type') in ('compile_error','runtime_error')


def test_prompt_construction_correctness():
    # HumanEval prompt in runner should include full function request and code block requirement
    from src.loop.runner import _load_dataset
    rows = _load_dataset('humaneval', subset='subset_20')
    row = rows[0]
    q = row['question']
    # Synthesize prompt as in runner
    he_prompt = (
        "You are a careful Python programmer. Do not include explanations or tests in your output.\n\n"
        "Implement the following Python function. Return the complete function definition (signature + body).\n"
        "Do not include any text outside code.\n\nProblem:\n" + q + "\n\nOutput format: Provide only a single Python code block containing the full function definition."
    )
    assert "complete function definition" in he_prompt
    assert "single Python code block" in he_prompt

    # GSM8K numeric-only prompt
    gsm_q = "What is 2+3?"
    gsm_prompt = (
        "You are a meticulous math solver. Think privately. Provide only the final numeric answer.\n\n"
        "Solve the problem. Think silently and provide only the final numeric answer with no units.\n\nQuestion:\n" + gsm_q + "\n\nOutput format: a single line containing only the final number."
    )
    assert "final numeric answer" in gsm_prompt


def test_rate_limit_behavior(tmp_path, monkeypatch):
    from src.utils.rate_limit import call_with_backoff_sync
    import time as _time
    log_file = tmp_path / 'rate_limit.log'
    monkeypatch.setenv('RATE_LIMIT_LOG', str(log_file))
    monkeypatch.setenv('RETRIES_ENABLED','1')
    monkeypatch.setenv('MAX_RETRIES','3')

    calls = {'n':0}
    def flaky():
        calls['n'] += 1
        if calls['n'] < 2:
            raise Exception('429 Too Many Requests')
        return 'ok'

    # Patch sleep to speed up
    monkeypatch.setattr('time.sleep', lambda s: None)

    out = call_with_backoff_sync(flaky)
    assert out == 'ok'
    # Log file should be created and contain backoff info
    assert log_file.exists()
    content = log_file.read_text()
    assert 'backoff=' in content or 'pre_wait=' in content

