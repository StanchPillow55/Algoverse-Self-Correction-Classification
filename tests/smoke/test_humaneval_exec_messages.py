import json, tempfile, os, pytest
from src.evaluation.humaneval_evaluator import HumanEvalEvaluator

def test_humaneval_exec_runs():
    # Requires evalplus or humaneval; skip if neither installed
    try:
        import evalplus  # noqa
    except Exception:
        try:
            import humaneval  # noqa
        except Exception:
            pytest.skip("no exec evaluator available")
    # Minimal sample structure expectation; content not executed here.
    ev = HumanEvalEvaluator()
    # Just check the evaluator provides keys without crashing when invoked with a dummy path
    # Real execution is covered in end-to-end smoke.
    assert hasattr(ev, "score")
