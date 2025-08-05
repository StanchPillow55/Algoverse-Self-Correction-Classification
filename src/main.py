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
from .workflow import ClassificationWorkflow
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
@click.option('--data-file', required=True, help='Path to dataset CSV file')
@click.option('--model-type', default='logistic_regression', 
              type=click.Choice(['logistic_regression', 'decision_tree']),
              help='Type of model to use')
@click.option('--output-dir', default='./outputs', help='Directory to save results')
def run_experiment(data_file: str, model_type: str, output_dir: str):
    """Run a complete classification experiment."""
    logger.info(f"Running classification experiment with {model_type}")
    
    try:
        workflow = ClassificationWorkflow()
        results = workflow.run_classification_experiment(
            dataset_path=data_file,
            model_type=model_type
        )
        
        # Save results
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        import json
        with open(output_path / f'experiment_results_{model_type}.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nExperiment Results ({model_type}):")
        print("=" * 40)
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"Training samples: {results['training_samples']}")
        print(f"Test samples: {results['test_samples']}")
        print(f"Features used: {results['feature_count']}")
        
        logger.info(f"Results saved to {output_path / f'experiment_results_{model_type}.json'}")
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        raise


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
    print("  - run-experiment: Run complete classification experiment")


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
