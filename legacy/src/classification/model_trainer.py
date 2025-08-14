"""
Model selection and training for LLM error classification.

Handles baseline model training and validation.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Trains and evaluates classification models for LLM errors."""

    def __init__(self, model_type: str = 'logistic_regression'):
        """
        Initialize the model trainer.

        Args:
            model_type: Type of model to train (`logistic_regression` or `decision_tree`).
        """
        self.model_type = model_type
        self.model = None

    def select_model(self) -> None:
        """Select the type of classification model based on user input."""
        if self.model_type == 'logistic_regression':
            self.model = LogisticRegression(max_iter=1000)
            logger.info("Using Logistic Regression model.")
        elif self.model_type == 'decision_tree':
            self.model = DecisionTreeClassifier(random_state=42)
            logger.info("Using Decision Tree model.")
        else:
            raise ValueError(f"Invalid model type: {self.model_type}")

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model on the provided dataset."""
        if not self.model:
            self.select_model()

        self.model.fit(X, y)
        logger.info("Model training complete.")

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Evaluate the model using the test data and return metrics."""
        predictions = self.model.predict(X)
        
        report = classification_report(y, predictions, output_dict=True)
        cm = confusion_matrix(y, predictions)
        accuracy = accuracy_score(y, predictions)

        logger.info("Model evaluation metrics computed.")
        return {
            "classification_report": report,
            "confusion_matrix": cm,
            "accuracy": accuracy
        }

    def train_and_evaluate(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Train and evaluate the model using the specified train/test split.

        Returns evaluation metrics and confusion matrix.
        """
        self.train(X_train, y_train)
        evaluation_results = self.evaluate(X_test, y_test)
        return evaluation_results

    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> float:
        """Perform cross-validation and return the average accuracy."""
        if not self.model:
            self.select_model()

        scores = cross_val_score(self.model, X, y, cv=cv)
        logger.info(f"Cross-validation accuracy: {np.mean(scores):.2f}")
        return np.mean(scores)

    def prepare_data(self, df: pd.DataFrame, feature_columns: list, label_column: str) -> Tuple[np.ndarray, ...]:
        """
        Prepare the feature matrix and labels for training/testing.

        Args:
            df: DataFrame containing features and labels.
            feature_columns: List of columns to use as features.
            label_column: Column to use as the label.

        Returns:
            Feature and label arrays for training and testing.
        """
        X = df[feature_columns].values
        y = df[label_column].values

        # Split data into train/test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logger.info("Data split into train and test sets.")
        return X_train, X_test, y_train, y_test

