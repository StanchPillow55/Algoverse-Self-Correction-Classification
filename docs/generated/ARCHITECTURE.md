# 🏗️ Architecture - Algoverse Self-Correction Classification

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    EXPERIMENT ORCHESTRATION                      │
│  run_full_scale_study.py / run_ensemble_experiments.py          │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                      DATASET MANAGER                             │
│  src/data/scaling_datasets.py                                    │
│  - Auto-download from HuggingFace                               │
│  - Deterministic subsets for reproducibility                    │
│  - GSM8K, HumanEval, SuperGLUE, MathBench                       │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                   TEACHER/LEARNER LOOP                           │
│  ┌─────────────┐      ┌─────────────┐      ┌─────────────┐      │
│  │   LEARNER   │ ───▶ │   TEACHER   │ ───▶ │    RTS      │      │
│  │ (Model API) │      │ (Bias Det.) │      │ (Templates) │      │
│  └─────────────┘      └─────────────┘      └─────────────┘      │
│        │                     │                    │              │
│        │                     ▼                    │              │
│        │              ┌─────────────┐             │              │
│        └─────────────▶│  FEEDBACK   │◀────────────┘              │
│                       │ (Coaching)  │                            │
│                       └─────────────┘                            │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                    ANALYSIS PIPELINE                             │
│  - Power-law fitting (Δ = A × ModelSize^α)                      │
│  - Cost tracking (USD per experiment)                           │
│  - Statistical significance testing                             │
└─────────────────────────────────────────────────────────────────┘
```

## Multi-Turn Self-Correction Process

```
Turn 0: Initial Answer
├── LearnerBot generates response + confidence
├── TeacherBot detects bias (overconfidence, error patterns)
├── Save reasoning trace
└── Evaluate accuracy

Turn 1: Self-Correction
├── RTS selects coaching template based on detected bias
├── LearnerBot re-attempts with bias-aware prompt
├── Save reasoning trace
└── Measure improvement (Δ accuracy)

Turn 2: Final Correction
├── Template selection based on confidence + previous improvement
├── Final answer generation
└── Compute total delta improvement
```

## Key Components

### LearnerBot (`src/agents/learner.py`)
- Multi-provider support (OpenAI, Anthropic, Replicate)
- Confidence scoring from model outputs
- Token usage tracking for cost analysis

### TeacherBot (`src/agents/teacher.py`)
- Bias detection:
  - Overconfidence: High confidence + wrong answer
  - Underconfidence: Low confidence + correct answer
  - Pattern errors: Systematic reasoning mistakes
  - Calculation errors: Arithmetic mistakes

### Reprompt Template Selection (`src/rts/policy.py`)
- Confidence-aware template selection
- Templates: Devils advocate, step-by-step, concise correction
- Error-type specific prompts

### Scaling Analysis (`src/scaling/analysis.py`)
- Power-law fitting: Δ = A × ModelSize^α
- R² and confidence intervals
- Cost-benefit threshold analysis

## Ensemble System

```
┌─────────────────────────────────────────────────────────┐
│                  ENSEMBLE VOTING                         │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐                 │
│  │ Model 1 │  │ Model 2 │  │ Model 3 │                 │
│  │ (GPT)   │  │(Claude) │  │ (Llama) │                 │
│  └────┬────┘  └────┬────┘  └────┬────┘                 │
│       │            │            │                       │
│       └────────────┼────────────┘                       │
│                    ▼                                    │
│            ┌──────────────┐                            │
│            │ VOTE AGGR.   │                            │
│            │ - Majority   │                            │
│            │ - Weighted   │                            │
│            │ - Consensus  │                            │
│            │ - Adaptive   │                            │
│            └──────────────┘                            │
└─────────────────────────────────────────────────────────┘
```

## Output Structure

```
full_scale_study_results/
├── full_scale_study_results.json    # Complete experiment metadata
├── cost_estimate.json               # Cost breakdown
├── csv_results/
│   └── analysis_dashboard.txt       # Statistical summary
└── reasoning_traces/                # Per-turn reasoning
    ├── math/{qid}/turn_{N}_reasoning.txt
    └── code/{qid}/turn_{N}_reasoning.txt
```

## Design Decisions

### Multi-Provider Abstraction
- **Why:** Compare models across providers fairly
- **How:** Unified interface in LearnerBot
- **Trade-off:** Provider-specific features may not be exposed

### Checkpointing System
- **Why:** Experiments can take hours, API failures happen
- **How:** Atomic writes, resumable from any checkpoint
- **Trade-off:** Additional I/O overhead

### Deterministic Subsets
- **Why:** Reproducibility across experiment runs
- **How:** Seeded random sampling with fixed seeds
- **Trade-off:** May not represent full dataset distribution
