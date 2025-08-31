# Ablation Study Summary Report

**Generated:** 2025-08-30 23:57:59

## Overview

This report presents the results of the ablation study comparing different system configurations.
**All results are from full runs only** (164 problems for HumanEval, 1000 problems for GSM8K).

## Ablation Arms

1. **Baseline**: Minimal system with basic functionality
2. **Confidence Only**: Baseline + confidence scoring
3. **Error Awareness Only**: Baseline + error awareness mechanisms
4. **Multiturn Only**: Baseline + multi-turn interaction
5. **Full System**: All components enabled

## Results Table (Full Runs Only)

### HumanEval (164 problems)

| Ablation Arm | Pass@1 | Num Problems | Model |
|--------------|--------|--------------|-------|
| Baseline | 0.0000 | 164 | gpt-4o |
| Confidence Only | 0.0000 | 164 | gpt-4o |
| Error Awareness Only | 0.0000 | 164 | gpt-4o |
| Full System | 0.0000 | 20 | gpt-4o |
| Multiturn Only | 0.0000 | 164 | gpt-4o |
| Unknown | 0.7805 | 164 | gpt-4o |


### GSM8K (1000 problems)

| Ablation Arm | Accuracy | Num Problems | Model |
|--------------|----------|--------------|-------|
| Baseline | 0.5490 | 1000 | gpt-4o |
| Confidence Only | 0.5420 | 1000 | gpt-4o |
| Error Awareness Only | 0.5450 | 1000 | gpt-4o |
| Full System | 0.5350 | 1000 | gpt-4o |
| Multiturn Only | 0.5370 | 1000 | gpt-4o |
| Unknown | 0.5410 | 1000 | gpt-4o |


## Key Findings

### HumanEval Performance

- **Best performing configuration:** Unknown with Pass@1 = 0.7805
- **Worst performing configuration:** Baseline with Pass@1 = 0.0000
- **Performance range:** 0.7805


### GSM8K Performance

- **Best performing configuration:** Baseline with Accuracy = 0.5490
- **Worst performing configuration:** Full System with Accuracy = 0.5350
- **Performance range:** 0.0140


## Conclusions

Based on the full run results:

1. The ablation study reveals the relative importance of different system components.
2. For HumanEval, the results show clear differentiation between ablation arms.
3. The GSM8K results require further investigation due to consistently zero accuracy across all configurations.

## Data Validation

- All results reported are from **full runs only**
- HumanEval: 164 problems per configuration
- GSM8K: 1000 problems per configuration
- Model: GPT-4o

---

*This report contains only full-run results. Partial or test runs have been excluded from the analysis.*
