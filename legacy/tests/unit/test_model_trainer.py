# Model trainer utility

import unittest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add src to path for testing
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.classification.model_trainer import ModelTrainer
except ImportError:
    import pytest
    pytest.skip("ModelTrainer not available", allow_module_level=True)

class TestModelTrainer(unittest.TestCase):

    def setUp(self):
        """Set up data for testing."""
        # Create fake data for testing
        self.df = pd.DataFrame({
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100),
            'label': np.random.randint(0, 2, size=100)
        })
        
        self.feature_columns = ['feature1', 'feature2']
        self.label_column = 'label'

    def test_train_and_evaluate_logistic_regression(self):
        """Test training and evaluating with logistic regression."""
        trainer = ModelTrainer('logistic_regression')
        X_train, X_test, y_train, y_test = trainer.prepare_data(self.df, self.feature_columns, self.label_column)
        results = trainer.train_and_evaluate(X_train, y_train, X_test, y_test)
        
        print("Accuracy:", results['accuracy'])  # For debugging purposes

    def test_train_and_evaluate_decision_tree(self):
        """Test training and evaluating with decision tree."""
        trainer = ModelTrainer('decision_tree')
        X_train, X_test, y_train, y_test = trainer.prepare_data(self.df, self.feature_columns, self.label_column)
        results = trainer.train_and_evaluate(X_train, y_train, X_test, y_test)
        
        print("Accuracy:", results['accuracy'])  # For debugging purposes

if __name__ == "__main__":
    unittest.main()
