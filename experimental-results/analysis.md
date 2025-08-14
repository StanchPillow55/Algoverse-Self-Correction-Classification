# Analysis Report

## Dataset & Setup
- **Dataset:** Math-20 sample (20 questions; subset of GitHub QnA)  
- **Models:** Learner = GPT-4o-mini, Teacher = GPT-4o-mini (OpenAI)  
- **Parameters:** `max_turns=3`, temperatures `0.2/0.0`, `max_tokens=40`  
- **Environment:** Python 3.12.7 (Anaconda), macOS ARM64, OpenAI v1.97.1, pandas 2.2.3  
- **Commit:** 2a16b1e81dc78821f3310f69a6daa6275ffdc92d  
(See `experiment_metadata.json` for full details.)

## Methods Compared
- **Teacher–Learner–RTS** (confidence-aware, bias detection, STOP rules)
- **GPT-4 self-correction baseline** (one targeted try)

## Key Results
- **Teacher–Learner**: Final accuracy **0.30** @ N=20 (`experiment_fresh_math20.json`).  
- **Validation run**: Final accuracy **0.30** @ N=20 (`teacher_learner_validation.json`).  
- **Baseline**: Final accuracy **0.15** @ N=20 (`fresh_baseline_math20.jsonl`).

**Improvement:** **+15 percentage points** (2× baseline).


## Auto-generated Metrics

| run | type | items | initial_acc | final_acc | delta_acc | mean_turns | acc_per_1k_tokens | error |
|---|---|---|---|---|---|---|---|---|
| experiment_fresh_math20.json | teacher_learner | 20 | 0.2 | 0.3 | 0.09999999999999998 | 2.5 |  |  |
| fresh_baseline_math20.jsonl | self_correction | 20 |  | 0.15 |  | 1.0 |  |  |
| teacher_learner_validation.json | teacher_learner | 20 | 0.2 | 0.3 | 0.09999999999999998 | 2.5 |  |  |
