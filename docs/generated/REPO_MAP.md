# 🗺️ Repo Map - Algoverse Self-Correction Classification

## Project Overview
ML research pipeline studying self-correction scaling laws across 7 LLM model sizes (1.8B-175B parameters), using teacher-learner dynamics, bias detection, and multi-turn correction cycles.

## Directory Structure

```
Algoverse-Self-Correction-Classification/
├── src/                              # Main source code
│   ├── agents/                       # Learner/Teacher bots
│   │   ├── learner.py               # LearnerBot: model interface
│   │   └── teacher.py               # Bias detection & feedback
│   ├── data/                         # Dataset management
│   │   ├── scaling_datasets.py      # Auto-download real datasets
│   │   ├── humaneval_loader.py      # HumanEval loader
│   │   └── gsm8k_loader.py          # GSM8K loader
│   ├── ensemble/                     # Multi-model voting
│   │   ├── runner.py                # Ensemble experiment runner
│   │   └── voting.py                # Voting strategies
│   ├── eval/                         # Evaluation
│   │   └── reasoning_extractor.py   # Extract reasoning traces
│   ├── rts/                          # Reprompt Template Selection
│   │   └── policy.py                # Template selection logic
│   ├── scaling/                      # Scaling analysis
│   │   ├── model_registry.py        # Model configs & costs
│   │   └── analysis.py              # Power-law fitting
│   ├── tools/                        # Tool-augmented QA
│   ├── utils/                        # Helpers
│   └── main.py                       # CLI entry point
│
├── configs/                          # Configuration files
│   ├── scaling_models.json          # Model definitions
│   ├── experiments/                 # Feature flag YAML configs
│   └── ensemble_experiments/        # Ensemble configs
│
├── data/                             # Dataset files
│   └── scaling/                     # Downloaded datasets
│
├── run_full_scale_study.py          # Full scaling study runner
├── run_ensemble_experiments.py      # Ensemble experiment runner
├── run_toolqa_experiments.py        # ToolQA experiments
│
├── full_scale_study_results/        # Experiment outputs
├── tracked_runs/                    # Individual run traces
│
├── paper/                            # Research paper drafts
│
└── docs/
    ├── ENSEMBLE_GUIDE.md            # Ensemble documentation
    ├── TOOLQA_GUIDE.md              # ToolQA documentation
    └── generated/                   # Generated docs
```

## Core Components

### Entry Points
| File | Purpose |
|------|---------|
| `src/main.py` | CLI for individual experiments |
| `run_full_scale_study.py` | Full scaling study orchestration |
| `run_ensemble_experiments.py` | Multi-model ensemble experiments |
| `run_toolqa_experiments.py` | Tool-augmented QA experiments |

### Agent System (`src/agents/`)
| File | Purpose |
|------|---------|
| `learner.py` | Model interface, confidence scoring, multi-provider |
| `teacher.py` | Bias detection, feedback generation |

### Data Management (`src/data/`)
| File | Purpose |
|------|---------|
| `scaling_datasets.py` | Auto-download from HuggingFace |
| `humaneval_loader.py` | HumanEval with code execution |
| `gsm8k_loader.py` | GSM8K math problems |

### Scaling Analysis (`src/scaling/`)
| File | Purpose |
|------|---------|
| `model_registry.py` | 7 models, parameter counts, API costs |
| `analysis.py` | Power-law fitting, statistical analysis |

## Datasets
| Dataset | Size | Type | Source |
|---------|------|------|--------|
| GSM8K | 1000 | Math word problems | HuggingFace |
| HumanEval | 164 | Code generation | HuggingFace |
| SuperGLUE | 1000 | Multi-task reasoning | HuggingFace |
| MathBench | 1000 | Advanced math | GitHub |

## Model Registry
| Model | Provider | Size | Cost/1K |
|-------|----------|------|---------|
| GPT-4o-mini | OpenAI | 1.8B | $0.00015 |
| Claude Haiku | Anthropic | 3.0B | $0.00025 |
| GPT-4o | OpenAI | 8.0B | $0.0025 |
| Claude Sonnet 3.5 | Anthropic | 70B | $0.003 |
| GPT-4 | OpenAI | 175B | $0.03 |
| Claude Opus | Anthropic | 175B | $0.015 |
