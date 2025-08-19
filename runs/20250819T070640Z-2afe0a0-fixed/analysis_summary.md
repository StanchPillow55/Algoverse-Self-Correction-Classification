# Full-Schema Trace Logging Experiment

## Configuration
- **Run ID:** 20250819T070640Z-2afe0a0-fixed
- **Git Commit:** 2afe0a0
- **Provider:** openai  
- **Model:** gpt-4o-mini
- **Demo Mode:** 0
- **Dataset:** data/math_sample_100.csv

## Trace Schema Implementation
This experiment implements comprehensive full-schema trace logging that captures:

### Per-Example Metadata
- Problem ID, text, dataset split
- Run ID, git commit, timestamps
- Final answer and correctness

### Per-Turn Details  
- **Prompts:** Complete conversation history and template instructions
- **Responses:** Model outputs and processed answers
- **Feedback:** Teacher/evaluator feedback and signals
- **Confidence:** Model-reported confidence scores
- **Correctness:** Turn-level accuracy evaluation
- **Latency:** Turn processing time in milliseconds
- **Signals:** Continue/stop evaluator decisions

### Files Generated
- `traces.jsonl` - Complete multi-turn conversation traces
- `run_config.json` - Experiment parameters
- `subset_20_summary.json` - 20-item results
- `subset_100_summary.json` - 100-item results

## Verification Results
The experiment verifies that:
1. ✅ Traces are generated in valid JSONL format
2. ✅ Full schema captures all required fields  
3. ✅ Results are not all zeros/false (accuracy validation)
4. ✅ Both subset_20 and subset_100 execute successfully

This replaces the previous minimal trace logger with comprehensive logging.
