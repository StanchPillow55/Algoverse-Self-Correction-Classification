# Deterministic Sampling System

This document explains the deterministic sampling system implemented to ensure reproducible and fair comparisons between multi-turn and ensemble experiments in the self-correction classification project.

## Overview

The deterministic sampling system ensures that:
1. **Identical problem subsets** are used across different experiment types (multi-turn vs ensemble)
2. **Reproducible results** are obtained when re-running experiments
3. **Fair comparisons** can be made between different models and approaches
4. **Consistent scaling studies** use nested subsets (smaller subsets are proper prefixes of larger ones)

## Key Components

### 1. Scripts Created

- **`run_anthropic_multiturn_experiments.py`** - Enhanced multi-turn experiment runner with deterministic sampling and early termination
- **`create_deterministic_subsets.py`** - Generates deterministic subsets for all datasets
- **`validate_deterministic_sampling.py`** - Validates that sampling is truly deterministic

### 2. Deterministic Sampling Implementation

#### ScalingDatasetManager (Enhanced)
Located in `src/data/scaling_datasets.py`, line 499-528:

```python
def load_dataset(self, dataset_name: str, sample_size: int = None, seed: int = 42) -> List[Dict[str, Any]]:
    """Load a dataset with optional deterministic sampling."""
    # ... 
    if sample_size and sample_size < len(samples):
        import random
        random.seed(seed)  # Set deterministic seed
        samples = random.sample(samples, sample_size)
        logger.info(f"Sampled {sample_size} from {len(data.get('samples', []))} samples using seed {seed}")
    return samples
```

#### Deterministic Subset Files
The preferred method creates deterministic subset files using **first-N sampling**:
- `data/scaling/{dataset}_deterministic_{size}.json`
- Uses first N samples from the original dataset (not random)
- Ensures perfect reproducibility across all experiments

#### Runner Integration
Both experiment runners support deterministic sampling:

**Multi-turn Runner** (`src/loop/runner.py`, lines 92-110):
```python
# Try deterministic subset first
deterministic_file = Path(f"data/scaling/gsm8k_deterministic_{subset_size}.json")
if deterministic_file.exists():
    print(f"📊 Loading deterministic GSM8K subset: {subset_size} samples (ensures reproducibility)")
    # Load from deterministic file
else:
    # Fallback to seeded sampling with consistent seed
    return dm.load_dataset('gsm8k', sample_size=int(subset_size), seed=42)
```

**Ensemble Runner** (`src/ensemble/runner.py`, lines 96-114):
```python
# Identical deterministic sampling logic for ensemble experiments
```

## Usage Instructions

### 1. Create Deterministic Subsets

First, generate deterministic subsets for your datasets:

```bash
# Create subsets for GSM8K and MathBench
python3 create_deterministic_subsets.py --datasets gsm8k,mathbench --sizes 100,500,1000

# Verify the subsets are created correctly
python3 create_deterministic_subsets.py --datasets gsm8k --sizes 100 --verify
```

### 2. Run Multi-turn Experiments with Deterministic Sampling

```bash
# Run Anthropic multi-turn experiments with deterministic sampling
python3 run_anthropic_multiturn_experiments.py \
    --datasets gsm8k,mathbench \
    --sample_sizes 100,500 \
    --models claude-3-haiku-20240307,claude-3-5-sonnet-20241210
```

### 3. Run Ensemble Experiments (Also Uses Deterministic Sampling)

```bash
# Ensemble experiments automatically use the same deterministic subsets
python3 -m src.ensemble.runner \
    --dataset gsm8k \
    --subset subset_100 \
    --provider anthropic \
    --model claude-3-haiku-20240307
```

### 4. Validate Deterministic Behavior

```bash
# Validate that sampling is truly deterministic
python3 validate_deterministic_sampling.py \
    --datasets gsm8k,mathbench \
    --sizes 100,500 \
    --output validation_report.json
```

## Validation Results

The validation system performs multiple tests:

1. **Determinism Validation**: Tests that multiple runs with the same seed produce identical samples
2. **Subset Files Validation**: Ensures deterministic subset files are properly formatted
3. **Nesting Validation**: Verifies that smaller subsets are proper prefixes of larger subsets
4. **Sampling Methods Comparison**: Compares different sampling approaches for consistency

Example validation output:
```
========================================================
VALIDATION SUMMARY
========================================================
Total tests: 15
Passed tests: 15
Success rate: 100.0%
Overall status: PASS

📊 GSM8K RESULTS:
  Determinism: 3/3 passed
  Subset files: 3/3 valid  
  Nesting: ✅ PASS

📊 MATHBENCH RESULTS:
  Determinism: 3/3 passed
  Subset files: 3/3 valid
  Nesting: ✅ PASS
```

## Benefits of This System

### 1. **Reproducible Research**
- Identical results when re-running experiments
- Version control of exact dataset subsets used
- Traceable experimental conditions

### 2. **Fair Comparisons**
- Multi-turn and ensemble experiments use identical problem sets
- No sampling bias between different approaches
- Consistent difficulty distribution across all experiments

### 3. **Efficient Experimentation**
- Early termination logic saves costs on poor-performing configurations
- Deterministic subsets enable rapid iteration
- Cost estimation before running expensive experiments

### 4. **Scaling Studies**
- Nested subsets enable proper scaling analysis
- 100-sample subset is always the first 100 samples of 500-sample subset
- Consistent problem ordering across all subset sizes

## Technical Implementation Details

### Deterministic Subset File Format

```json
{
  "name": "GSM8K",
  "description": "Deterministic subset of GSM8K",
  "original_dataset": "gsm8k",
  "sampling_method": "deterministic_first_n",
  "sample_size": 100,
  "total_original_size": 8792,
  "created_by": "create_deterministic_subsets.py",
  "samples": [
    {
      "id": "0",
      "question": "Natalie's apple orchard has 4060 apple trees...",
      "answer": "420",
      "split": "train",
      "difficulty": "grade_school",
      "topic": "math_word_problem"
    }
    // ... 99 more samples
  ]
}
```

### Sampling Priority Order

1. **Deterministic files** (preferred): `data/scaling/{dataset}_deterministic_{size}.json`
2. **Seeded sampling** (fallback): `ScalingDatasetManager.load_dataset(seed=42)`
3. **Default loading** (last resort): Original dataset loaders

### Random Seed Standardization

- **Default seed**: 42 (used across all experiments)
- **Seed consistency**: Same seed used for multi-turn and ensemble experiments
- **Seed isolation**: Each dataset uses independent seeding to prevent cross-contamination

## Future Enhancements

### 1. **Multi-dataset Cross-validation**
- Implement cross-dataset consistency checks
- Ensure consistent difficulty ordering across datasets

### 2. **Stratified Sampling**
- Add support for stratified sampling by difficulty/topic
- Maintain determinism while improving sample distribution

### 3. **Experiment Lineage Tracking**
- Track which experiments used which exact subsets
- Enable retroactive analysis of experimental conditions

### 4. **Automated Subset Generation**
- Automatically generate subsets when new datasets are added
- Validate subset quality and distribution

## Troubleshooting

### Common Issues

1. **Missing Deterministic Files**
   ```bash
   ⚠️ Deterministic subset not found, using seeded sampling for GSM8K subset_100
   ```
   **Solution**: Run `create_deterministic_subsets.py` to generate missing files

2. **Inconsistent Sample Counts**
   ```bash
   ❌ Sample count mismatch for gsm8k size 100: 95 != 100
   ```
   **Solution**: Regenerate deterministic subsets with `--force` flag

3. **Validation Failures**
   ```bash
   ❌ Hash mismatch between run 1 and run 2
   ```
   **Solution**: Check for non-deterministic data loading or random state issues

### Debug Mode

Enable debug logging to trace sampling behavior:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Integration with Existing Systems

### Experiment Runners
- ✅ Multi-turn runner (`src/loop/runner.py`)
- ✅ Ensemble runner (`src/ensemble/runner.py`) 
- 🔄 Analysis scripts automatically detect deterministic experiments

### Dataset Loaders
- ✅ GSM8K loader with deterministic subset support
- ✅ HumanEval loader (inherently deterministic)
- ✅ ScalingDatasetManager with seeded fallback

### Output Systems
- ✅ CSV formatters preserve dataset metadata
- ✅ Trace formatters include sampling method information
- ✅ Analysis dashboards show deterministic vs non-deterministic experiments

---

This deterministic sampling system ensures that all experimental comparisons in this project are fair, reproducible, and scientifically sound. The multi-layered approach (deterministic files + seeded fallback) provides robustness while maintaining perfect reproducibility for critical comparisons between multi-turn and ensemble approaches.