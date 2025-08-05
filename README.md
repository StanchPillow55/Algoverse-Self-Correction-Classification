# LLM Error Classification Pipeline

A comprehensive pipeline for identifying and correcting errors in Large Language Model (LLM) outputs. This project implements a classification system that can semantically analyze model outputs and route errors to appropriate re-prompting strategies.

## Inspiration

This work is inspired by the paper "Understanding the Dark Side of LLMs Intrinsic Self-Correction" (Sharma et al., 2023), which provides foundational understanding of error typology in self-correcting LLMs.

## Error Types Supported

The pipeline can identify and classify the following error types:

- **Answer Wavering**: Model changes answers without clear justification
- **Prompt Bias**: Responses heavily influenced by prompt framing
- **Overthinking**: Unnecessarily complex solutions to simple problems  
- **Cognitive Overload**: Struggles with complex multi-step reasoning
- **Perfectionism Bias**: Over-corrects or provides unnecessarily perfect solutions

## Pipeline Architecture

### 1. Data Collection
- Curate/generate datasets with annotated examples of different error types
- Initial synthetic generation using LLMs (GPT-4, Claude Sonnet, Llama 3) to bootstrap labeled datasets

### 2. Semantic Embedding
- Generate embeddings via LLMs, Sentence Transformers, or OpenAI embeddings
- Optional integration of logits from Llama models

### 3. Classification Model
- Logistic Regression or Decision Trees initially (interpretable, baseline models)
- Multi-class classification to handle different error types

### 4. Post-Processing
- Route model predictions to specific re-prompting strategies
- Apply appropriate corrections based on error type and confidence

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

## Usage

### Command Line Interface

The pipeline provides a CLI for various operations:

```bash
# Show pipeline information
python -m src.main info

# Generate synthetic training data
python -m src.main generate-data --samples-per-error 20

# Train the classification model
python -m src.main train --data-file data/synthetic/synthetic_errors.csv

# Predict error type for text
python -m src.main predict --text "Your text here"
```

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
