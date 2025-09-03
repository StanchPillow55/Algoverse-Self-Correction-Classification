import os
from pathlib import Path
from src.loop.runner import run_dataset

def test_tracing_writer_creates_run_dir(tmp_path, monkeypatch):
    # Use demo mode to avoid API calls
    monkeypatch.setenv("DEMO_MODE", "1")
    monkeypatch.setenv("PROVIDER", "demo")
    monkeypatch.setenv("RUN_ID", "smoke_test")
    monkeypatch.setenv("OPENAI_TEMPERATURE", "0.2")
    monkeypatch.setenv("SEEDS", "1,2,3")

    out = tmp_path / "traces.json"
    # Run small GSM8K-like dataset
    o = run_dataset("data/smoke/gsm8k16.csv", str(out), max_turns=2, provider="demo")
    assert o["summary"]["items"] > 0

    # Check that run directory under runs/ exists with metrics and traces
    runs_dir = Path("runs")
    assert runs_dir.exists()
    found = list(runs_dir.glob("*__*__*__*__seed*__t*__mt*"))
    assert found, "Expected structured run directory to be created"
    rd = found[0]
    assert (rd / "metrics.json").exists()
    assert (rd / "traces.json").exists()

