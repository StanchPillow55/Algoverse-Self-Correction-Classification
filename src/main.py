"""
Main entry point for the LLM Error Classification Pipeline.

This module orchestrates the entire pipeline from data collection to classification.
"""

import click
import logging
from pathlib import Path
from .data_collection.synthetic_generator import SyntheticDataGenerator
from .embeddings.embedding_generator import EmbeddingGenerator
from .classification.classifier import ErrorClassifier
from .utils.config import Config
from .utils.error_types import get_all_error_types

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.group()
def cli():
    """LLM Error Classification Pipeline"""
    pass


@cli.command()
@click.option('--output-dir', default='./data/synthetic', help='Output directory for synthetic data')
@click.option('--samples-per-error', default=10, help='Number of samples per error type')
def generate_data(output_dir: str, samples_per_error: int):
    """Generate synthetic training data."""
    logger.info("Starting synthetic data generation...")
    
    # Load configuration
    Config.load()
    
    # Initialize generator
    generator = SyntheticDataGenerator()
    
    # Get default prompts
    prompts = generator.get_default_prompts()
    
    # Generate dataset
    dataset = generator.generate_dataset(prompts, samples_per_error)
    
    # Save dataset
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    dataset.to_csv(output_path / 'synthetic_errors.csv', index=False)
    logger.info(f"Generated {len(dataset)} samples and saved to {output_path / 'synthetic_errors.csv'}")


@cli.command()
@click.option('--data-file', required=True, help='Path to training data CSV file')
@click.option('--model-dir', default='./models', help='Directory to save trained model')
def train(data_file: str, model_dir: str):
    """Train the error classification model."""
    logger.info(f"Starting model training with data from {data_file}")
    
    # This is a skeleton - actual implementation would:
    # 1. Load the dataset
    # 2. Generate embeddings
    # 3. Train the classifier
    # 4. Save the trained model
    
    logger.info("Training complete (skeleton implementation)")


@cli.command()
@click.option('--text', required=True, help='Text to classify')
@click.option('--model-dir', default='./models', help='Directory with trained model')
def predict(text: str, model_dir: str):
    """Predict error type for given text."""
    logger.info(f"Predicting error type for: {text[:50]}...")
    
    # This is a skeleton - actual implementation would:
    # 1. Load the trained model
    # 2. Generate embeddings for the input text
    # 3. Make prediction
    # 4. Return error type and confidence
    
    logger.info("Prediction complete (skeleton implementation)")


@cli.command()
def info():
    """Show information about the pipeline."""
    error_types = get_all_error_types()
    
    print("LLM Error Classification Pipeline")
    print("=" * 40)
    print(f"Supported error types: {len(error_types)}")
    for et in error_types:
        print(f"  - {et.value}")
    
    print("\nAvailable commands:")
    print("  - generate-data: Create synthetic training data")
    print("  - train: Train the classification model")
    print("  - predict: Classify text for error types")


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
