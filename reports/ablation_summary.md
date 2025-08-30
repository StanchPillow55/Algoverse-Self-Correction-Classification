# Ablation Study Results

## GSM8K Results

| arm                  |   final_accuracy |   turn_0 |      turn_1 |      turn_2 |
|:---------------------|-----------------:|---------:|------------:|------------:|
| baseline             |            0.145 |    0.145 | nan         | nan         |
| confidence_only      |            0.15  |    0.15  | nan         | nan         |
| error_awareness_only |            0.15  |    0.15  | nan         | nan         |
| full_system          |            0.165 |    0.135 |   0.0273973 |   0.0144928 |
| multiturn_only       |            0.185 |    0.15  |   0.0285714 |   0.0227273 |

### Delta from Baseline

| Arm | Final Accuracy | Delta |
|-----|---------------|-------|
| baseline | 0.290 | +0.000 |
| baseline | 0.000 | -0.290 |
| confidence_only | 0.300 | +0.010 |
| confidence_only | 0.000 | -0.290 |
| error_awareness_only | 0.300 | +0.010 |
| error_awareness_only | 0.000 | -0.290 |
| multiturn_only | 0.000 | -0.290 |
| multiturn_only | 0.370 | +0.080 |
| full_system | 0.000 | -0.290 |
| full_system | 0.330 | +0.040 |

## HUMANEVAL Results

| arm                  |   final_accuracy |   turn_0 |
|:---------------------|-----------------:|---------:|
| baseline             |             0.95 |     0.95 |
| confidence_only      |             0.9  |     0.9  |
| error_awareness_only |             0.95 |     0.95 |
| full_system          |             0.95 |     0.95 |
| multiturn_only       |             0.95 |     0.95 |

### Delta from Baseline

| Arm | Final Accuracy | Delta |
|-----|---------------|-------|
| baseline | 0.950 | +0.000 |
| baseline | 0.950 | +0.000 |
| confidence_only | 0.900 | -0.050 |
| error_awareness_only | 0.950 | +0.000 |
| multiturn_only | 0.950 | +0.000 |
| full_system | 0.950 | +0.000 |
| full_system | 0.950 | +0.000 |

## Overall Summary

- **Baseline Performance**: Foundation for comparison
- **Confidence Only**: Impact of confidence scoring
- **Error Awareness Only**: Impact of error detection
- **Multiturn Only**: Impact of iterative refinement
- **Full System**: Combined effect of all components

## Key Findings

- Best performing arm: **full_system** with average accuracy 0.557
- Multiturn improvement over baseline: -0.108
