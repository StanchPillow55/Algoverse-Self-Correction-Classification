# HumanEval Integration Guide

This document explains how to use the HumanEval code generation dataset with the self-correction classification system.

## Overview

The system supports HumanEval (Human-Eval) code generation benchmarks alongside existing math reasoning datasets. HumanEval tests models' ability to generate correct Python functions based on natural language descriptions.

## Key Features

### 1. Dataset Loading
- **Automatic loading**: The system automatically downloads and loads the HumanEval dataset from the OpenAI repository
- **Subset support**: Full dataset (164 problems), subset_100 (100 problems), or subset_20 (20 problems)
- **Fallback data**: Built-in demo problems when the official dataset is unavailable

### 2. Code Execution and Evaluation
- **Safe execution**: Code is executed in sandboxed environments with timeouts
- **Test-based scoring**: Uses the official HumanEval test suites for each problem
- **Pass@K metrics**: Supports standard pass@1, pass@K evaluation
- **Detailed execution results**: Captures stdout, stderr, runtime, and error information

### 3. Integration with Self-Correction
- **Specialized prompting**: Uses code-specific prompts for function generation
- **Single-turn focus**: HumanEval typically uses single-turn evaluation (no multi-turn self-correction)
- **Error analysis**: Supports confidence scoring and bias detection for code generation

## Usage Examples

### 1. Basic HumanEval Run
```bash
# Run demo mode (no API calls needed)
python scripts/run_humaneval_sample.py --demo --subset=3

# Run with OpenAI API on 20 problems
python scripts/run_humaneval_sample.py --subset=20

# Run full dataset with GPT-4
python scripts/run_humaneval_sample.py --subset=full --model=gpt-4o
```

### 2. Configuration File

Create a YAML configuration file:

```yaml
# configs/experiments/my_humaneval.yaml
name: "HumanEval Self-Correction Experiment"
dataset:
  name: "humaneval"
  subset: "subset_20"
model:
  provider: "openai"
  name: "gpt-4o-mini"
  temperature: 0.2
experiment:
  max_turns: 1
  features:
    enable_confidence: true
    enable_error_awareness: true
    enable_multi_turn: false
```

### 3. Direct Python Usage

```python
from src.loop.runner import run_dataset

# Run HumanEval experiment
result = run_dataset(
    dataset_csv="humaneval",
    traces_out="outputs/my_humaneval_traces.json",
    max_turns=1,
    provider="openai",
    model="gpt-4o-mini",
    subset="subset_20",
    config={
        'features': {
            'enable_confidence': True,
            'enable_error_awareness': True,
            'enable_multi_turn': False
        }
    },
    experiment_id="my_humaneval_experiment",
    dataset_name="humaneval"
)

print(f"Accuracy: {result['summary']['final_accuracy_mean']:.3f}")
```

## Output Format

### Trace Structure
Each HumanEval problem generates a trace with:

```json
{
  "qid": "HumanEval/0",
  "question": "def has_close_elements(numbers: List[float], threshold: float) -> bool:",
  "reference": "",
  "turns": [
    {
      "answer": "def has_close_elements(numbers, threshold):\n    ...",
      "response_text": "Full model response including reasoning",
      "self_conf": 0.85,
      "teacher_bias": "None",
      "teacher_conf": 0.7,
      "template": null,
      "accuracy": 1,
      "execution_details": {
        "passed": true,
        "passed_count": 7,
        "total_count": 7,
        "stdout": "All tests passed",
        "stderr": "",
        "runtime_ms": 45.2,
        "error": ""
      }
    }
  ],
  "final_accuracy": 1
}
```

### Key Fields

- **qid**: HumanEval problem identifier (e.g., "HumanEval/0")
- **question**: Function signature and docstring
- **answer**: Generated function code
- **accuracy**: 1 if all tests pass, 0 otherwise
- **execution_details**: Detailed code execution results
  - `passed`: Boolean indicating overall success
  - `passed_count/total_count`: Test statistics
  - `stdout/stderr`: Execution output
  - `runtime_ms`: Execution time
  - `error`: Any execution errors

## Architecture Details

### Components

1. **Data Loader** (`src/data/humaneval_loader.py`)
   - Downloads official HumanEval dataset
   - Normalizes data format
   - Provides demo fallback data

2. **Code Scorer** (`src/eval/humaneval_scorer.py`)
   - Extracts functions from model responses
   - Executes code safely with timeouts
   - Runs official test suites
   - Returns detailed execution results

3. **Code Executor** (`src/eval/code_executor.py`)
   - Sandboxed Python execution
   - Security restrictions and timeouts
   - Capture stdout/stderr/exceptions

4. **Integration** (`src/loop/runner.py`)
   - Detects HumanEval tasks by `topic` field
   - Uses specialized code prompts
   - Handles pass@K sampling
   - Integrates with tracing system

### Data Flow

```
HumanEval Dataset → Data Loader → Runner → Model → Code Scorer → Execution Results
                                    ↓
                             Trace Logger → JSON/CSV Output
```

### Prompting Strategy

For HumanEval tasks, the system uses specialized prompts:

```
"You are a careful Python programmer. Do not include explanations or tests in your output.

Implement the following Python function. Return the complete function definition (signature + body).
Do not include any text outside code.

Problem:
[Function signature and docstring]

Output format: Provide only a single Python code block containing the full function definition."
```

## Environment Variables

- **DEMO_MODE**: Set to "1" to use demo mode (no API calls)
- **OPENAI_API_KEY**: Required for OpenAI models
- **ANTHROPIC_API_KEY**: Required for Anthropic models
- **PASS_K**: Number of samples for pass@K evaluation (default: 1)

## Dataset Information

- **Total problems**: 164 hand-crafted programming problems
- **Languages**: Python 3
- **Difficulty**: Entry-level to intermediate programming tasks
- **Topics**: String processing, math, data structures, algorithms
- **Evaluation**: Functional correctness via test cases

## Comparison with GSM8K

| Aspect | HumanEval | GSM8K |
|--------|-----------|--------|
| Domain | Code generation | Math word problems |
| Output | Python functions | Numeric answers |
| Evaluation | Test execution | Exact match |
| Multi-turn | Typically single | Benefits from multi-turn |
| Metrics | Pass@1, Pass@K | Exact match accuracy |

## Best Practices

1. **Single-turn evaluation**: HumanEval typically uses one attempt per problem
2. **Temperature settings**: Use low temperature (0.0-0.2) for deterministic code generation
3. **Timeout handling**: Set appropriate execution timeouts (10-30 seconds)
4. **Error analysis**: Use execution details to understand failure modes
5. **Pass@K sampling**: For research, consider multiple samples per problem

## Troubleshooting

### Common Issues

1. **Dataset not loading**: Check internet connection, ensure git is installed
2. **Code execution errors**: Verify Python environment, check security restrictions
3. **API rate limits**: Use appropriate delays between requests
4. **Memory issues**: Use subset datasets for initial testing

### Debug Mode

Enable detailed logging:

```bash
export DEMO_MODE=1  # Use demo mode for testing
export DEBUG=1      # Enable debug output
python scripts/run_humaneval_sample.py --demo --subset=3
```

## Performance Expectations

Based on existing literature:

- **GPT-4**: ~70-80% pass@1 on HumanEval
- **GPT-3.5**: ~40-50% pass@1 on HumanEval  
- **Code-specific models**: Often higher performance
- **Self-correction**: Limited benefit for code tasks vs. math reasoning

The self-correction system provides detailed analysis of model behavior, confidence calibration, and error modes even when multi-turn correction has limited benefit for code generation tasks.