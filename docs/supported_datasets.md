# Supported Datasets Overview

This document provides a comprehensive overview of all datasets currently supported by the self-correction classification system.

## Summary of Supported Datasets

The system currently supports **4 major dataset types**:

| Dataset | Domain | Problems | Evaluation Method | Multi-turn | Status |
|---------|--------|----------|------------------|------------|--------|
| **GSM8K** | Math reasoning | 8,500 | Exact match | ✅ Beneficial | ✅ Full support |
| **HumanEval** | Code generation | 164 | Test execution | ⚠️ Limited benefit | ✅ Full support |
| **ToolQA** | Tool reasoning | 100 samples | Custom metrics | ✅ Beneficial | ✅ Supported |
| **Custom CSV** | Flexible | Variable | Configurable | ✅ Configurable | ✅ Full support |

## Dataset Details

### 1. GSM8K (Grade School Math 8K)
- **Description**: Mathematical word problems requiring multi-step reasoning
- **Size**: 8,500 training problems, subset options available
- **Format**: Natural language questions → Numeric answers
- **Evaluation**: Exact match accuracy after parsing
- **Multi-turn**: Highly beneficial for self-correction
- **Usage**: Primary benchmark for math reasoning evaluation

**Example Problem**:
```
Question: "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast every morning..."
Answer: "18"
```

**How to Use**:
```python
from src.data.gsm8k_loader import load_gsm8k_dataset
data = load_gsm8k_dataset()
```

### 2. HumanEval
- **Description**: Hand-crafted programming problems for Python code generation
- **Size**: 164 problems total
- **Format**: Function signature + docstring → Complete Python function
- **Evaluation**: Functional correctness via test execution
- **Multi-turn**: Limited benefit (single-turn preferred)
- **Usage**: Standard benchmark for code generation

**Example Problem**:
```python
def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers, are any two numbers closer 
    to each other than given threshold. """
```

**How to Use**:
```python
from src.data.humaneval_loader import load_humaneval_dataset
data = load_humaneval_dataset(subset="subset_20")
```

### 3. ToolQA
- **Description**: Questions requiring tool usage and reasoning
- **Size**: ~100 sample problems
- **Format**: Natural language questions → Tool-assisted answers
- **Evaluation**: Custom metrics based on tool usage correctness
- **Multi-turn**: Beneficial for iterative tool usage
- **Usage**: Research on tool-augmented reasoning

**Example Problem**:
```
Question: "What is the population of the capital city of France?"
Expected: Use location tool → population tool → answer
```

### 4. Custom CSV Datasets
- **Description**: Flexible format supporting any Q&A task
- **Size**: Variable based on input file
- **Format**: CSV with configurable column mapping
- **Evaluation**: Exact match or custom scoring
- **Multi-turn**: Configurable based on task type
- **Usage**: Custom experiments and new dataset integration

**Required Columns**: `qid`, `question`, `ground_truth` (or similar)

**How to Use**:
```python
# Any CSV file with question-answer pairs
result = run_dataset(
    dataset_csv="path/to/your_dataset.csv",
    # ... other parameters
)
```

## Quick Start Guide

### Running GSM8K Experiments
```bash
# Math reasoning with multi-turn self-correction
python src/main.py --dataset=data/scaling/gsm8k_sample.csv --max-turns=3
```

### Running HumanEval Experiments  
```bash
# Code generation (typically single-turn)
python scripts/run_humaneval_sample.py --subset=20 --provider=openai
```

### Running ToolQA Experiments
```bash
# Tool-augmented reasoning
python src/main.py --dataset=data/scaling/toolqa_sample_100.csv --max-turns=2
```

### Running Custom Dataset
```bash
# Any CSV with qid, question, ground_truth columns
python src/main.py --dataset=your_data.csv --max-turns=3
```

## Configuration Examples

### GSM8K Configuration
```yaml
dataset:
  name: "gsm8k"
  subset: "sample"
experiment:
  max_turns: 3
  features:
    enable_multi_turn: true
    enable_error_awareness: true
model:
  provider: "openai"
  name: "gpt-4o-mini"
```

### HumanEval Configuration
```yaml
dataset:
  name: "humaneval" 
  subset: "subset_20"
experiment:
  max_turns: 1
  features:
    enable_multi_turn: false
    enable_confidence: true
model:
  provider: "openai"
  name: "gpt-4o-mini"
  temperature: 0.2
```

## System Architecture

### Dataset Loading Pipeline
```
Dataset Request → Auto-Detection → Specialized Loader → Normalization → Runner
```

**Auto-Detection Logic**:
- "humaneval" in name → HumanEval loader
- "gsm8k" in name → GSM8K loader  
- "toolqa" in name → ToolQA handling
- ".csv" extension → Generic CSV loader

### Evaluation Pipeline
```
Model Response → Task-Specific Scorer → Execution/Matching → Accuracy Score
```

**Scoring Methods**:
- **GSM8K**: Exact match after numeric parsing
- **HumanEval**: Code execution against test suites
- **ToolQA**: Custom tool usage metrics
- **CSV**: Exact match (default) or custom

## Performance Benchmarks

Based on the repository's experimental results:

### GSM8K Results (Sample runs)
- **GPT-4o**: ~85-90% accuracy with self-correction
- **GPT-4o-mini**: ~75-80% accuracy with self-correction  
- **Multi-turn improvement**: 10-15% gain typical

### HumanEval Results (Expected)
- **GPT-4o**: ~70-80% pass@1
- **GPT-4o-mini**: ~50-60% pass@1
- **Multi-turn improvement**: Limited (2-5% typical)

## Adding New Datasets

To add a new dataset, follow these steps:

### 1. Create Data Loader
```python
# src/data/your_dataset_loader.py
def load_your_dataset(subset=None):
    # Load and normalize data
    return [{"qid": "...", "question": "...", "ground_truth": "..."}]
```

### 2. Add Scorer (if needed)
```python
# src/eval/your_dataset_scorer.py  
def score_your_dataset(task, response):
    # Custom scoring logic
    return {"accuracy": 1 or 0, "details": {...}}
```

### 3. Update Runner
```python
# In src/loop/runner.py _load_dataset function
elif "your_dataset" in dataset_csv.lower():
    return load_your_dataset(subset)
```

### 4. Add Evaluation Logic
```python
# In src/loop/runner.py run_dataset function
if is_your_dataset:
    acc = your_dataset_accuracy(task, answer)
```

## Environment Setup

### Required API Keys
```bash
export OPENAI_API_KEY="your_openai_key"
export ANTHROPIC_API_KEY="your_anthropic_key"
```

### Optional Configuration
```bash
export DEMO_MODE=1          # Use demo mode (no API calls)
export DEBUG=1              # Enable debug output
export PASS_K=1             # Pass@K evaluation for code
export MAX_CONCURRENT=1     # Concurrent requests limit
```

### Python Dependencies
The system requires standard scientific Python packages:
- `pandas`, `numpy`: Data handling
- `openai`, `anthropic`: LLM APIs
- `subprocess`, `ast`: Code execution (HumanEval)
- `json`, `csv`: Data I/O

## Future Dataset Integration

The architecture is designed to easily support additional datasets:

### Planned/Potential Datasets
- **MATH**: Advanced mathematical problems
- **MBPP**: More Python programming problems  
- **CommonsenseQA**: Commonsense reasoning
- **StrategyQA**: Multi-hop reasoning
- **Code contests**: Competitive programming problems

### Integration Requirements
1. **Data loader**: Normalize to standard format
2. **Evaluation**: Define accuracy metrics
3. **Prompting**: Task-appropriate prompt templates
4. **Multi-turn policy**: Determine if multi-turn helps

The system's modular design allows rapid integration of new datasets while maintaining consistency in evaluation and self-correction analysis.