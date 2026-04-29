# 🧪 Testing - Algoverse Self-Correction Classification

## Testing Approach

### Demo Mode Testing
```bash
# No API keys required
export DEMO_MODE=1 PROVIDER=demo

# Quick system test
python -m src.main info
python -m src.main run --dataset humaneval --subset subset_20 --max-turns 3 \
  --out runs/demo/test.json --provider demo
```

### Validation Runs
```bash
# Small-scale validation with real APIs (cost: ~$5)
python run_full_scale_study.py --mode validation \
  --datasets gsm8k --models gpt-4o-mini,claude-haiku
```

### Production Runs
```bash
# Full scaling study (cost: ~$200)
python run_full_scale_study.py --mode production \
  --datasets all --models all
```

## Test Coverage

### Individual Components

**Dataset Loaders:**
```bash
# Test dataset download and loading
python -c "
from src.data.scaling_datasets import ScalingDatasetManager
dm = ScalingDatasetManager()
dm.download_dataset('gsm8k')
print('✅ GSM8K loaded')
"
```

**Model Registry:**
```bash
# Test cost estimation
python -c "
from src.scaling.model_registry import estimate_experiment_cost
cost = estimate_experiment_cost('gpt-4o', 100, avg_tokens_per_sample=2000)
print(f'Estimated: \${cost[\"total_cost_usd\"]:.2f}')
"
```

**Power-Law Analysis:**
```bash
# Test statistical analysis
python -c "
from src.scaling.analysis import fit_power_law
import numpy as np
sizes = np.array([1.8e9, 3e9, 8e9, 70e9, 175e9])
improvements = np.array([0.05, 0.08, 0.12, 0.18, 0.22])
result = fit_power_law(sizes, improvements)
print(f'Scaling exponent: {result.scaling_exponent:.3f}')
"
```

### Ensemble System
```bash
# Demo ensemble test
python run_ensemble_experiments.py --dataset gsm8k --subset subset_20 --demo

# Analyze ensemble results
python -m src.ensemble.metrics outputs/ensemble_experiments/*/traces.json outputs/analysis
```

## Completed Experiment Results

### High-Quality Runs (from `CURRENT_RESEARCH_STATUS.md`)

| Run | Model | Dataset | Accuracy | Status |
|-----|-------|---------|----------|--------|
| `fullscale_gpt-4o-mini_gsm8k_*` | GPT-4o-mini | GSM8K | **87.5%** | ✅ Complete |
| `fullscale_gpt-4o-mini_humaneval_*` | GPT-4o-mini | HumanEval | **82.3%** | ✅ Complete |
| `fullscale_claude-haiku_gsm8k_*` | Claude-Haiku | GSM8K | **56.7%** | ✅ Complete |
| `fullscale_claude-haiku_humaneval_*` | Claude-Haiku | HumanEval | **45.7%** | ✅ Complete |

### Standardized Analysis
- **119 experimental runs** extracted and analyzed
- **12 models tested**
- **9 datasets** covered

## Checkpointing System

The pipeline includes robust checkpointing for experiment resumption:

```bash
# Run with checkpointing
python -m src.main run --dataset gsm8k --checkpoint-every 10

# Resume interrupted experiment
python -m src.main run --dataset gsm8k --resume
```

**Features:**
- Atomic writes prevent corruption
- Resumable from any checkpoint
- Error recovery with automatic retry

## Known Limitations

1. **API Quota:** OpenAI/Anthropic rate limits can interrupt experiments
2. **HumanEval Execution:** Code runs locally (sandboxed but not isolated)
3. **Large Files:** SuperGLUE dataset is 364MB
4. **Cost:** Full study costs ~$200

## CI/CD Status
**Not Configured** - Research codebase, experiments run manually with careful cost control.
