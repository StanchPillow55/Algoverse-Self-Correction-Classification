"""
Decision Tree classifier for predicting LLM error types.

Provides functionality to train a model and predict errors from text embeddings.
"""

from sklearn.tree import DecisionTreeClassifier
from typing import List, Any, Dict
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ErrorClassifier:
    """Decision Tree classifier for LLM error types."""

    def __init__(self):
        """Initialize the classifier with a decision tree model."""
        self.model = DecisionTreeClassifier(random_state=42)
        logger.info("Initialized ErrorClassifier with Decision Tree model.")

    def train(self, X: List[List[float]], y: List[int]) -> None:
        """Train the classifier on the provided data."""
        self.model.fit(X, y)
        logger.info("Training complete.")

    def predict(self, X: List[List[float]]) -> List[int]:
        """Predict the error types for a batch of inputs."""
        return self.model.predict(X).tolist()

    def predict_proba(self, X: List[List[float]]) -> List[List[float]]:
        """Get the class probabilities for a batch of inputs."""
        return self.model.predict_proba(X).tolist()

    def get_feature_importances(self) -> Dict[str, float]:
        """Get the feature importance scores from the trained model."""
        feature_importances = self.model.feature_importances_
        return {f"feature_{i}": importance for i, importance in enumerate(feature_importances)}

