"""
Workflow orchestrator for the LLM error classification pipeline.

Integrates data preparation, preprocessing, embedding generation, 
model training, and evaluation into a complete workflow.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging

from .data_collection.preprocessor import DataPreprocessor
from .data_collection.dataset_curator import DatasetCurator
from .utils.error_types import ErrorType, get_all_error_types
from .utils.config import Config

logger = logging.getLogger(__name__)


class ClassificationWorkflow:
    """Orchestrates the complete classification workflow."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the workflow with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        Config.load(config_path or '.env')
        
        self.preprocessor = DataPreprocessor()
        self.curator = DatasetCurator()
        
        # Label mapping for error types
        self.error_types = get_all_error_types()
        self.label_to_id = {error_type.value: i for i, error_type in enumerate(self.error_types)}
        self.id_to_label = {i: error_type.value for i, error_type in enumerate(self.error_types)}
        
        logger.info("Initialized ClassificationWorkflow")
    
    def prepare_data(self, dataset_path: str, text_column: str = 'response', 
                    label_column: str = 'error_type') -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Prepare data for training by loading, validating, and preprocessing.
        
        Args:
            dataset_path: Path to the dataset CSV file
            text_column: Name of the text column to process
            label_column: Name of the label column
            
        Returns:
            Processed dataset and preparation statistics
        """
        logger.info(f"Loading dataset from {dataset_path}")
        
        # Load dataset
        dataset = pd.read_csv(dataset_path)
        
        # Validate dataset
        validation_results = self.curator.validate_dataset(dataset)
        if not validation_results['is_valid']:
            raise ValueError(f"Dataset validation failed: {validation_results['errors']}")
        
        # Clean dataset
        dataset = self.curator.clean_dataset(dataset)
        
        # Preprocess text
        dataset = self.preprocessor.preprocess_dataset(dataset, text_column)
        
        # Convert labels to numeric IDs
        if label_column in dataset.columns:
            dataset['label_id'] = dataset[label_column].map(self.label_to_id)
            # Remove rows with unknown labels
            dataset = dataset.dropna(subset=['label_id'])
            dataset['label_id'] = dataset['label_id'].astype(int)
        
        stats = {
            "total_samples": len(dataset),
            "label_distribution": dataset[label_column].value_counts().to_dict() if label_column in dataset.columns else {},
            "validation_results": validation_results
        }
        
        logger.info(f"Data preparation complete: {len(dataset)} samples prepared")
        return dataset, stats
    
    def create_feature_matrix(self, texts: list) -> np.ndarray:
        """
        Create feature matrix from text data.
        
        For this skeleton implementation, we'll use simple text statistics
        as features. In a full implementation, this would use embeddings.
        
        Args:
            texts: List of text strings
            
        Returns:
            Feature matrix (n_samples, n_features)
        """
        features = []
        
        for text in texts:
            # Simple text statistics as features
            text_features = [
                len(text),  # Text length
                len(text.split()),  # Word count
                text.count('?'),  # Question marks
                text.count('!'),  # Exclamation marks
                text.count(','),  # Commas
                len(set(text.split())),  # Unique words
                text.count('but'),  # Hedging words
                text.count('however'),
                text.count('maybe'),
                text.count('perhaps'),
            ]
            features.append(text_features)
        
        feature_matrix = np.array(features)
        logger.info(f"Created feature matrix with shape {feature_matrix.shape}")
        return feature_matrix
    
    def run_classification_experiment(self, dataset_path: str, 
                                    model_type: str = 'logistic_regression',
                                    text_column: str = 'response',
                                    label_column: str = 'error_type') -> Dict[str, Any]:
        """
        Run a complete classification experiment.
        
        Args:
            dataset_path: Path to the dataset
            model_type: Type of model to use
            text_column: Name of text column
            label_column: Name of label column
            
        Returns:
            Experiment results including metrics and model info
        """
        logger.info(f"Starting classification experiment with {model_type}")
        
        # Prepare data
        dataset, prep_stats = self.prepare_data(dataset_path, text_column, label_column)
        
        if len(dataset) == 0:
            raise ValueError("No valid samples after data preparation")
        
        # Create features
        texts = dataset[f'{text_column}_cleaned'].tolist()
        X = self.create_feature_matrix(texts)
        y = dataset['label_id'].values
        
        # Split data
        from sklearn.model_selection import train_test_split
        
        # Check if we have enough samples for stratification
        unique_classes, class_counts = np.unique(y, return_counts=True)
        min_class_count = np.min(class_counts)
        n_classes = len(unique_classes)
        
        # Disable stratification if dataset is too small
        stratify = y if min_class_count >= 2 and len(y) >= n_classes * 2 else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.4, random_state=42, stratify=stratify
        )
        
        # Train model (simplified for skeleton)
        if model_type == 'logistic_regression':
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(max_iter=1000, random_state=42)
        elif model_type == 'decision_tree':
            from sklearn.tree import DecisionTreeClassifier
            model = DecisionTreeClassifier(random_state=42)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Evaluate
        from sklearn.metrics import accuracy_score, classification_report
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        results = {
            "model_type": model_type,
            "data_stats": prep_stats,
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "feature_count": X.shape[1],
            "accuracy": accuracy,
            "classification_report": report,
            "feature_names": [
                "text_length", "word_count", "question_marks", "exclamation_marks",
                "commas", "unique_words", "but_count", "however_count", 
                "maybe_count", "perhaps_count"
            ]
        }
        
        logger.info(f"Experiment complete. Accuracy: {accuracy:.4f}")
        return results
    
    def get_workflow_info(self) -> Dict[str, Any]:
        """Get information about the workflow configuration."""
        return {
            "error_types": [et.value for et in self.error_types],
            "label_mapping": self.label_to_id,
            "preprocessor_config": {
                "lowercase": self.preprocessor.lowercase,
                "remove_punctuation": self.preprocessor.remove_punctuation,
                "min_length": self.preprocessor.min_length
            }
        }
