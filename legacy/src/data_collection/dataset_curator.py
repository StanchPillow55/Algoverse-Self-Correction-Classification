"""
Dataset curator for managing and validating training data.

Handles dataset cleaning, validation, and preparation for training.
"""

import pandas as pd
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class DatasetCurator:
    """Curate and validate datasets for training."""
    
    def __init__(self):
        """Initialize the dataset curator."""
        logger.info("Initialized DatasetCurator")
    
    def validate_dataset(self, dataset: pd.DataFrame) -> Dict[str, Any]:
        """Validate dataset format and content."""
        validation_results = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "stats": {
                "total_samples": len(dataset),
                "error_type_distribution": {}
            }
        }
        
        # Basic validation checks
        required_columns = ["prompt", "response", "error_type"]
        missing_columns = [col for col in required_columns if col not in dataset.columns]
        
        if missing_columns:
            validation_results["is_valid"] = False
            validation_results["errors"].append(f"Missing columns: {missing_columns}")
        
        if "error_type" in dataset.columns:
            validation_results["stats"]["error_type_distribution"] = dataset["error_type"].value_counts().to_dict()
        
        logger.info(f"Dataset validation complete: {'PASSED' if validation_results['is_valid'] else 'FAILED'}")
        return validation_results
    
    def clean_dataset(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare dataset for training."""
        cleaned = dataset.copy()
        
        # Remove duplicates
        initial_size = len(cleaned)
        cleaned = cleaned.drop_duplicates()
        final_size = len(cleaned)
        
        if initial_size != final_size:
            logger.info(f"Removed {initial_size - final_size} duplicate samples")
        
        return cleaned
