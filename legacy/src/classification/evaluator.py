"""
Classification evaluator for computing metrics and analysis.

Provides comprehensive evaluation metrics and visualization tools.
"""

from sklearn.metrics import (
    classification_report, confusion_matrix, precision_recall_fscore_support,
    accuracy_score, precision_score, recall_score, f1_score
)
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class ClassificationEvaluator:
    """Evaluates classification models with comprehensive metrics."""

    def __init__(self, label_names: Optional[List[str]] = None):
        """
        Initialize the evaluator.
        
        Args:
            label_names: Names of the class labels for better reporting
        """
        self.label_names = label_names or ["no_error", "answer_wavering", "prompt_bias", 
                                          "overthinking", "cognitive_overload", "perfectionism_bias"]

    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """
        Compute comprehensive classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
            y_true, y_pred, average=None
        )
        
        # Classification report
        class_report = classification_report(y_true, y_pred, output_dict=True)
        
        # Confusion matrix
        conf_matrix = confusion_matrix(y_true, y_pred)
        
        metrics = {
            "accuracy": accuracy,
            "precision_weighted": precision,
            "recall_weighted": recall,
            "f1_weighted": f1,
            "precision_per_class": precision_per_class.tolist(),
            "recall_per_class": recall_per_class.tolist(),
            "f1_per_class": f1_per_class.tolist(),
            "support_per_class": support.tolist(),
            "classification_report": class_report,
            "confusion_matrix": conf_matrix.tolist(),
            "num_samples": len(y_true),
            "num_classes": len(np.unique(y_true))
        }
        
        logger.info(f"Computed metrics for {len(y_true)} samples across {len(np.unique(y_true))} classes")
        return metrics

    def analyze_misclassifications(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                 sample_texts: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze misclassified samples to understand error patterns.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            sample_texts: Optional list of original texts for analysis
            
        Returns:
            Dictionary with misclassification analysis
        """
        misclassified_indices = np.where(y_true != y_pred)[0]
        
        analysis = {
            "total_misclassified": len(misclassified_indices),
            "misclassification_rate": len(misclassified_indices) / len(y_true),
            "misclassified_by_true_class": {},
            "misclassified_by_predicted_class": {},
            "confusion_pairs": []
        }
        
        # Analyze by true class
        for true_class in np.unique(y_true):
            mask = (y_true == true_class) & (y_true != y_pred)
            misclassified_count = np.sum(mask)
            total_count = np.sum(y_true == true_class)
            
            analysis["misclassified_by_true_class"][str(true_class)] = {
                "count": int(misclassified_count),
                "total": int(total_count),
                "rate": float(misclassified_count / total_count) if total_count > 0 else 0.0
            }
        
        # Analyze by predicted class
        for pred_class in np.unique(y_pred):
            mask = (y_pred == pred_class) & (y_true != y_pred)
            misclassified_count = np.sum(mask)
            total_predicted = np.sum(y_pred == pred_class)
            
            analysis["misclassified_by_predicted_class"][str(pred_class)] = {
                "count": int(misclassified_count),
                "total": int(total_predicted),
                "rate": float(misclassified_count / total_predicted) if total_predicted > 0 else 0.0
            }
        
        # Find most common confusion pairs
        for i in misclassified_indices:
            pair = (int(y_true[i]), int(y_pred[i]))
            analysis["confusion_pairs"].append(pair)
        
        # Count confusion pairs
        from collections import Counter
        confusion_counts = Counter(analysis["confusion_pairs"])
        analysis["most_common_confusions"] = confusion_counts.most_common(5)
        
        logger.info(f"Analyzed {len(misclassified_indices)} misclassified samples")
        return analysis

    def create_evaluation_report(self, y_true: np.ndarray, y_pred: np.ndarray, 
                               model_name: str = "Model") -> str:
        """
        Create a comprehensive evaluation report as a formatted string.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model being evaluated
            
        Returns:
            Formatted evaluation report string
        """
        metrics = self.compute_metrics(y_true, y_pred)
        misclass_analysis = self.analyze_misclassifications(y_true, y_pred)
        
        report = f"""
========================================
{model_name} Evaluation Report
========================================

Overall Metrics:
- Accuracy: {metrics['accuracy']:.4f}
- Weighted Precision: {metrics['precision_weighted']:.4f}
- Weighted Recall: {metrics['recall_weighted']:.4f}
- Weighted F1-Score: {metrics['f1_weighted']:.4f}

Dataset Information:
- Total Samples: {metrics['num_samples']}
- Number of Classes: {metrics['num_classes']}
- Misclassification Rate: {misclass_analysis['misclassification_rate']:.4f}

Per-Class Performance:
"""
        
        for i, (precision, recall, f1, support) in enumerate(zip(
            metrics['precision_per_class'],
            metrics['recall_per_class'], 
            metrics['f1_per_class'],
            metrics['support_per_class']
        )):
            class_name = self.label_names[i] if i < len(self.label_names) else f"Class_{i}"
            report += f"- {class_name}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}, Support={support}\n"
        
        report += f"\nMost Common Misclassifications:\n"
        for (true_class, pred_class), count in misclass_analysis['most_common_confusions']:
            true_name = self.label_names[true_class] if true_class < len(self.label_names) else f"Class_{true_class}"
            pred_name = self.label_names[pred_class] if pred_class < len(self.label_names) else f"Class_{pred_class}"
            report += f"- {true_name} → {pred_name}: {count} times\n"
        
        return report

    def save_evaluation_results(self, metrics: Dict[str, Any], filepath: str) -> None:
        """Save evaluation results to a file."""
        # Convert numpy arrays to lists for JSON serialization
        serializable_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                serializable_metrics[key] = value.tolist()
            else:
                serializable_metrics[key] = value
        
        import json
        with open(filepath, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)
        
        logger.info(f"Evaluation results saved to {filepath}")
