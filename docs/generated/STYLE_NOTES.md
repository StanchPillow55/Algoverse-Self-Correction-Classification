# 🎨 Style Notes - Algoverse Self-Correction Classification

## Language & Ecosystem
- **Primary Language:** Python 3.10+
- **Framework:** Custom ML research pipeline
- **APIs:** OpenAI, Anthropic, HuggingFace, Replicate
- **Data:** HuggingFace Datasets for GSM8K, HumanEval, SuperGLUE

## Code Conventions

### Module Structure
```
src/
├── agents/          # Learner and Teacher bots
├── analysis/        # Statistical analysis tools
├── data/            # Dataset loaders and managers
├── ensemble/        # Multi-model voting system
├── eval/            # Evaluation and metrics
├── loop/            # Main experiment loop
├── rts/             # Reprompt template selection
├── scaling/         # Model registry and cost tracking
├── tools/           # Tool-augmented QA
└── utils/           # Helpers, formatters, logging
```

### Naming Conventions
- **Functions:** `snake_case` (e.g., `detect_bias`, `fit_power_law`)
- **Classes:** `PascalCase` (e.g., `LearnerBot`, `ScalingDatasetManager`)
- **Constants:** `UPPER_SNAKE_CASE` (e.g., `MAX_RETRIES`, `DEMO_MODE`)
- **Config files:** lowercase with underscores (e.g., `scaling_models.json`)

### Configuration Patterns
- Environment variables via `.env` file with `python-dotenv`
- JSON configs in `configs/` directory
- YAML configs for experiment features (`configs/experiments/`)
- CLI args via `argparse` in main entry points

### Docstrings
- Module-level docstrings explaining purpose
- Function docstrings with Args/Returns
- Type hints on public functions

## Error Handling
- Graceful API error handling with retries
- Checkpointing for experiment resumption
- Demo mode fallback when API unavailable

## Research Code Patterns
- Reproducibility via seeded random sampling
- Comprehensive trace logging (per-turn reasoning)
- Cost tracking with real-time estimates
- Multi-provider abstraction layer

## Testing
- Manual test scripts in root directory
- `DEMO_MODE=1` for API-free testing
- Validation runs before production experiments
