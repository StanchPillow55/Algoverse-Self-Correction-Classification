from src.loop.runner import run_dataset

def test_demo_runner_math20(tmp_path):
    # run with demo mode; no external API calls
    out_file = tmp_path / "traces.json"
    res = run_dataset("data/math20.csv", str(out_file), max_turns=2, provider="demo")
    assert "summary" in res and "traces" in res
    assert res["summary"]["items"] >= 3
    # in demo, arithmetic should mostly be correct
    assert 0.5 <= res["summary"]["final_accuracy_mean"] <= 1.0
