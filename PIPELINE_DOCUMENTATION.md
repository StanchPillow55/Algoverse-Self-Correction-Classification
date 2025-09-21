# Complete Pipeline Documentation

## System Verification Status ✅

### 1. API Keys: ✅ FUNCTIONAL
- **OpenAI API**: Working (gpt-4o-mini, gpt-4o, gpt-3.5-turbo)
- **Anthropic API**: Working (claude-3-haiku, claude-3-sonnet, claude-3.5-sonnet)

### 2. Checkpointing: ✅ FUNCTIONAL
- Resume capability: Working
- Error handling: Working
- Atomic writes: Working
- Concurrent safety: Working

### 3. Ensemble Methods: ✅ FUNCTIONAL
- Multi-model voting: Working
- Heterogeneous ensembles: Working
- Demo mode available: Working

### 4. Dataset Loading: ✅ DETERMINISTIC
- **Deterministic subsets**: First-N samples for consistency
- **Seeded sampling**: Reproducible random sampling (seed=42)
- **Available datasets**: GSM8K (8,792), HumanEval (164), MathBench (100), SuperGLUE (1,000)

### 5. End-to-End: ✅ WORKING
- Full pipeline tested with 3 GSM8K questions
- Multi-turn self-correction functioning
- All output formats generated

---

## Pipeline Architecture

### Core Components

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Dataset       │───▶│   LearnerBot     │───▶│   Bias          │
│   Loader        │    │   (Multi-turn)   │    │   Detector      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Checkpoint    │    │   Template       │    │   Output        │
│   Manager       │    │   Engine         │    │   Formatter     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

---

## Exact Usage Instructions

### Main Entry Points

#### 1. Individual Experiments
```bash
python -m src.main run \
  --dataset gsm8k \
  --subset subset_100 \
  --provider openai \
  --model gpt-4o-mini \
  --out my_experiment \
  --max-turns 3 \
  --checkpoint-every 10
```

#### 2. Ensemble Experiments
```bash
python run_ensemble_experiments.py \
  --config configs/ensemble_experiments/heterogeneous_ensemble.json \
  --dataset gsm8k \
  --subset subset_50 \
  --output-dir outputs/ensemble_results
```

#### 3. Full Scale Studies
```bash
python run_full_scale_study.py \
  --models gpt-4o-mini claude-3-haiku \
  --datasets gsm8k humaneval \
  --output full_scale_results
```

---

## Detailed Pipeline Flow

### Step 1: Dataset Loading

**Script**: `src/data/scaling_datasets.py`

**Process**:
1. Load dataset from `data/scaling/{dataset}.json`
2. Apply subset selection:
   - **Deterministic**: `create_deterministic_subsets()` - Uses first N samples
   - **Random**: `load_dataset(seed=42)` - Reproducible random sampling

**Deterministic Subsets Available**:
```bash
# Created automatically:
data/scaling/gsm8k_deterministic_20.json
data/scaling/gsm8k_deterministic_50.json
data/scaling/gsm8k_deterministic_100.json
data/scaling/gsm8k_deterministic_500.json
data/scaling/gsm8k_deterministic_1000.json
```

### Step 2: Learner Bot Initialization

**Script**: `src/agents/learner.py`

**Configuration**:
```python
learner = LearnerBot(
    provider="openai",           # openai, anthropic, demo
    model="gpt-4o-mini",        # specific model name
    temperature=0.2,            # consistent across runs
    confidence_threshold=0.6,   # trigger multi-turn if below
    max_tokens=1024
)
```

### Step 3: Multi-Turn Processing Loop

**Script**: `src/loop/runner.py`

**Flow for Each Question**:
```python
# Turn 0: Initial response
prompt = base_prompt_template.format(question=question)
response, confidence = learner.answer(prompt)
accuracy = evaluate_answer(response, ground_truth)

# Bias Detection
bias_type, bias_conf = detect_bias(question, response, execution_result)

# Multi-turn Logic
if confidence < threshold or bias_detected:
    # Select correction template
    template = select_template(bias_type, accuracy, turn_number)
    
    # Turn 1: Self-correction attempt
    correction_prompt = template.format(
        original_response=response,
        feedback=bias_feedback
    )
    new_response, new_confidence = learner.answer(correction_prompt)
    new_accuracy = evaluate_answer(new_response, ground_truth)
```

### Step 4: Template System

**Location**: `prompts/templates/`

**Core Templates**:

#### Base Template (`base.txt`)
```
Question: {question}

Solve this step by step, showing your reasoning clearly.
```

#### Devils Advocate (`devils_advocate_v1.txt`)
```
Your previous response was:
{original_response}

Please reconsider your solution. Look for potential errors in:
1. Mathematical calculations
2. Logic assumptions
3. Reading comprehension

Provide a revised solution if needed.
```

#### Try Again Concise (`try_again_concise.txt`)
```
Previous answer: {original_response}

This appears incorrect. Please try again with a more careful approach.
Focus on accuracy over explanation length.
```

### Step 5: Bias Detection

**Script**: `src/bias/code_bias_detector.py` and `src/bias/math_bias_detector.py`

**Bias Types Detected**:
- **Confirmation Bias**: Over-confidence in initial answer
- **Anchoring Bias**: Stuck on first approach
- **Availability Bias**: Using most obvious solution

**Detection Process**:
```python
def detect_bias(question, response, execution_result):
    # Pattern matching
    confidence_indicators = extract_confidence_markers(response)
    
    # Error analysis
    if execution_result.get('error'):
        return analyze_error_patterns(error)
    
    # Content analysis
    reasoning_quality = assess_reasoning_depth(response)
    
    return bias_type, confidence_score
```

### Step 6: Checkpointing

**Script**: `src/utils/checkpoint.py`

**Atomic Write Process**:
```python
def append_result_atomic(result):
    # Add metadata
    result.update({
        "checkpoint_time": time.time(),
        "checkpoint_datetime": datetime.now().isoformat()
    })
    
    # Atomic write with file locking
    with open(checkpoint_file, "a") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        f.write(json.dumps(result) + "\n")
        f.flush()
        os.fsync(f.fileno())
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
```

**Resume Logic**:
```python
completed_ids = load_completed_ids()
for question in dataset:
    if question.id not in completed_ids:
        process_question(question)
        checkpoint_manager.append_result_atomic(result)
```

### Step 7: Output Generation

**Multiple Formats Generated**:

#### 1. Main Traces (`runs/{timestamp}/traces.json`)
```json
{
  "meta": {
    "model": "gpt-4o-mini",
    "dataset": "gsm8k",
    "max_turns": 3
  },
  "items": [
    {
      "id": "1824",
      "turns": [
        {
          "turn_index": 0,
          "confidence": 0.6,
          "accuracy": 1,
          "response_text": "...",
          "evaluator_feedback": {}
        }
      ],
      "final": {
        "predicted": "375",
        "correct": true
      }
    }
  ]
}
```

#### 2. CSV Results (`csv_results/`)
- `{dataset}_{model}_results_{timestamp}.csv` - Per-question details
- `{dataset}_{model}_summary_{timestamp}.csv` - Aggregated metrics
- `turn_analysis_{timestamp}.csv` - Multi-turn statistics

#### 3. Enhanced Traces (`outputs/enhanced_traces/`)
- `{experiment}_accuracy_data.json` - Detailed accuracy analysis
- `{experiment}_full_traces/` - Individual question files
- `{experiment}_summary_metrics.json` - High-level performance

---

## Key Configuration Files

### Dataset Configurations
```bash
# Main datasets
data/scaling/gsm8k.json          # 8,792 math problems
data/scaling/humaneval.json      # 164 code problems  
data/scaling/mathbench.json      # 100 advanced math
data/scaling/superglue.json      # 1,000 NLP tasks
```

### Ensemble Configurations
```bash
configs/ensemble_experiments/
├── demo_ensemble.json           # Demo mode (no API)
├── heterogeneous_ensemble.json  # Mixed providers
├── openai_basic.json           # OpenAI models only
└── anthropic_ensemble.json     # Anthropic models only
```

### Environment Variables
```bash
# Required API keys (loaded from .env or environment)
OPENAI_API_KEY=sk-proj-...
ANTHROPIC_API_KEY=sk-ant-api...
```

---

## Common Command Patterns

### Quick Tests
```bash
# Smoke test with 3 questions
python -m src.main run --dataset gsm8k --subset subset_3 --provider openai --model gpt-4o-mini --out smoke_test

# Demo mode (no API calls)
python -m src.main run --dataset gsm8k --subset subset_5 --provider demo --out demo_test
```

### Production Runs
```bash
# Full GSM8K with checkpointing
python -m src.main run --dataset gsm8k --provider openai --model gpt-4o-mini --out gsm8k_full --max-turns 3 --checkpoint-every 50

# Resume interrupted experiment
python -m src.main run --dataset gsm8k --provider openai --model gpt-4o-mini --out gsm8k_full --resume
```

### Ensemble Experiments
```bash
# Single ensemble config
python run_ensemble_experiments.py --config configs/ensemble_experiments/heterogeneous_ensemble.json --dataset gsm8k --subset subset_100

# Batch ensemble experiments
python run_ensemble_experiments.py --batch --configs-dir configs/ensemble_experiments --dataset gsm8k --subset subset_50
```

---

## Output Structure

```
outputs/
├── csv_results/                 # CSV format results
├── enhanced_traces/             # Detailed trace analysis
└── ensemble_experiments/       # Ensemble results

runs/
└── {timestamp}__{dataset}__{mode}__{model}__seed{N}__t{temp}__mt{turns}/
    ├── traces.json             # Main results
    ├── config.json             # Experiment config
    ├── metrics.json            # Summary metrics
    ├── structured_traces/      # Individual problem files
    └── prompts/               # Used prompt templates
```

---

## Quality Assurance

### Verification Commands
```bash
# Test API connectivity
python test_api_keys.py

# Test checkpointing system
python test_checkpoint.py --test=all

# Test dataset consistency
python test_dataset_consistency.py

# Test ensemble functionality
python run_ensemble_experiments.py --demo --config configs/ensemble_experiments/demo_ensemble.json --dataset gsm8k --subset subset_3
```

### Troubleshooting

**Common Issues**:
1. **"Unknown provider specified"** → Check API keys in environment
2. **"Deterministic subset not found"** → Creates seeded random subset automatically
3. **Checkpoint conflicts** → Uses file locking, should resolve automatically
4. **Memory issues** → Use `--checkpoint-every 10` for frequent saves

---

## Next Steps for Experiments

With the verified system, you can now:

1. **Run production experiments** with confidence
2. **Scale to full datasets** using checkpointing
3. **Compare ensemble methods** across different configurations
4. **Reproduce results** using deterministic subsets
5. **Analyze bias patterns** with comprehensive traces

The pipeline is ready for large-scale self-correction studies! 🚀