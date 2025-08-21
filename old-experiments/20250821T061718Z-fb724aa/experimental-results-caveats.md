# Known Caveats and Deviations

## Dataset Deviations
- **Original Dataset Size**: Used Math-20 sample instead of full 1364 questions for cost/time constraints
- **Question Selection**: First 20 questions from cached dataset (not randomized sample)
- **URL Branch**: Used `feat/initial-error-table` branch data instead of main branch due to 404 errors

## Model/API Constraints
- **Temperature Settings**: Learner used 0.2 temperature (not explicitly tuned per research proposal)
- **Max Tokens**: Limited to 40 tokens per response to control costs
- **Rate Limiting**: No explicit rate limiting implemented; relied on OpenAI default limits
- **Determinism**: Limited by OpenAI API; exact reproduction may vary due to model updates

## Implementation Status
- **Bias Detection**: Currently dominated by "Confirmation" bias (88% of cases)
- **Template Effectiveness**: Devils advocate template shows low success rate (6.7%)
- **Confidence Calibration**: Self-confidence vs teacher confidence not well discriminated
- **Stop Rules**: Basic implementation; more sophisticated rules not yet implemented

## Infrastructure Compromises
- **Error Handling**: Some API errors fall back to "0" responses
- **Logging**: Debug logging basic; not all edge cases captured
- **Caching**: Local caching may affect reproducibility across runs
- **Branch State**: Used development branch with some experimental code paths

## Known Issues Fixed During Experiment
- **LearnerBot Parameter Bug**: Fixed `tmpl` vs `template` parameter mismatch
- **CSV Loading**: Fixed URL→raw conversion and column mapping issues
- **Import Paths**: Fixed module import issues in test files

## Statistical Limitations
- **Small Sample Size**: N=20 may not capture full performance distribution
- **No Cross-Validation**: Single run without error bars or confidence intervals
- **Bias in Dataset**: Mathematical word problems may not represent general case
- **Cherry-Picking Risk**: Results based on specific subset of problems

## Reproducibility Notes
- **Git Commit**: Results from commit 2a16b1e (post-debugging fixes)
- **Environment**: macOS ARM64 with Anaconda Python 3.12.7
- **API Version**: OpenAI Python client v1.97.1 (results may vary with newer versions)
- **Timing**: Experiments run 2025-08-14, model behavior may change over time
