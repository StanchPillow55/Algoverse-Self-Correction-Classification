# Confidence-Aware Reprompt Selection for Reliable LLM Self-Correction (Teacher-Bot / Learner-Bot Edition)

This repo implements a *classification-free* teacher/learner loop. The **teacher-bot** semantically labels broad failure modes (Anchoring, Confirmation, Availability/Bandwagon, Hindsight, Overgeneralization), gates on confidence, and selects a **reprompt template**. The **learner-bot** attempts answers; the teacher iterates until STOP rules.

> Note: Legacy classifier files remain during this pivot and are skipped in tests.

## Installation

```bash
# Clone the repository
git clone https://github.com/StanchPillow55/Algoverse-Self-Correction-Classification.git
cd Algoverse-Self-Correction-Classification

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Configuration

Copy the example environment file and configure your API keys:

```bash
cp .env.example .env
# Edit .env with your API keys
```

## Quickstart
```bash
pip install -r requirements.txt
export DEMO_MODE=1
python -m src.main info
python -m src.main run --dataset data/math20.csv --max-turns 2
pytest -q
```

See `configs/run.yaml`, `rts_templates.json`, and `configs/stop_rules.yaml` for controls.

### Outputs
- `outputs/traces.json` — per-item turns with bias labels, confidence, templates, and accuracy deltas.

### Programmatic Usage

```python
from src.data_collection.synthetic_generator import SyntheticDataGenerator
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.classification.classifier import ErrorClassifier

# Generate synthetic data
generator = SyntheticDataGenerator()
dataset = generator.generate_dataset(prompts, samples_per_error=10)

# Generate embeddings
embedder = EmbeddingGenerator()
embeddings = embedder.generate_embeddings(texts)

# Train classifier
classifier = ErrorClassifier()
classifier.train(embeddings, labels)
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run smoke tests only
pytest tests/smoke

# Run with coverage
pytest --cov=src
```

## Project Structure

```
├── src/
│   ├── data_collection/          # Synthetic data generation
│   ├── embeddings/               # Text embedding generation
│   ├── classification/           # Error classification models
│   ├── post_processing/          # Error routing and correction
│   └── utils/                    # Utilities and configuration
├── tests/
│   ├── smoke/                    # Basic functionality tests
│   ├── unit/                     # Unit tests
│   └── integration/              # Integration tests
├── data/                         # Data storage
├── models/                       # Trained models
├── configs/                      # Configuration files
└── notebooks/                    # Jupyter notebooks for analysis
```

## Development Status

This is currently a research prototype implementing the core pipeline structure. Key components are in place with skeleton implementations that can be extended with full functionality.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Datasets via GitHub URLs

The pipeline can now run directly from the ground-truth CSVs hosted on GitHub.

### Run teacher–learner pipeline on the QnA URL:
```bash
# This requires OPENAI_API_KEY to be set in .env
python scripts/run_experiments.py --config configs/experiments/exp_qna_urls.yaml
```

### Run GPT-4 self-correction evaluator:
```bash
scripts/run_self_correction_eval.sh
```
