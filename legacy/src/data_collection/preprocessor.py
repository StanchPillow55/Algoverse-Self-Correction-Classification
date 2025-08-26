"""
Data preprocessor for text normalization and preparation.

Handles text cleaning, normalization, and preparation for embedding generation.
"""

import re
import string
from typing import List, Dict, Any
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Preprocessor for cleaning and normalizing text data."""
    
    def __init__(self, 
                 lowercase: bool = True,
                 remove_punctuation: bool = False,
                 remove_extra_whitespace: bool = True,
                 min_length: int = 10):
        """
        Initialize the preprocessor with configuration options.
        
        Args:
            lowercase: Convert text to lowercase
            remove_punctuation: Remove punctuation marks
            remove_extra_whitespace: Remove extra whitespace and normalize
            min_length: Minimum text length to keep
        """
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_extra_whitespace = remove_extra_whitespace
        self.min_length = min_length
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize a single text string."""
        if not isinstance(text, str):
            return ""
            
        # Remove extra whitespace
        if self.remove_extra_whitespace:
            text = re.sub(r'\s+', ' ', text.strip())
        
        # Convert to lowercase
        if self.lowercase:
            text = text.lower()
            
        # Remove punctuation
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
            
        return text.strip()
    
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """Preprocess a batch of texts."""
        processed_texts = []
        
        for text in texts:
            cleaned = self.clean_text(text)
            
            # Filter by minimum length
            if len(cleaned) >= self.min_length:
                processed_texts.append(cleaned)
            else:
                processed_texts.append("")  # Keep empty for alignment
                
        logger.info(f"Preprocessed {len(texts)} texts")
        return processed_texts
    
    def preprocess_dataset(self, dataset: pd.DataFrame, text_column: str = 'response') -> pd.DataFrame:
        """Preprocess texts in a dataset DataFrame."""
        df = dataset.copy()
        
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in dataset")
            
        # Preprocess the text column
        df[f'{text_column}_cleaned'] = self.preprocess_batch(df[text_column].tolist())
        
        # Remove rows with empty cleaned text
        initial_size = len(df)
        df = df[df[f'{text_column}_cleaned'] != ""]
        final_size = len(df)
        
        if initial_size != final_size:
            logger.info(f"Filtered out {initial_size - final_size} texts below minimum length")
            
        return df
    
    def get_preprocessing_stats(self, original_texts: List[str], processed_texts: List[str]) -> Dict[str, Any]:
        """Get statistics about the preprocessing step."""
        stats = {
            "original_count": len(original_texts),
            "processed_count": len([t for t in processed_texts if t]),
            "average_length_before": sum(len(t) for t in original_texts) / len(original_texts) if original_texts else 0,
            "average_length_after": sum(len(t) for t in processed_texts if t) / len([t for t in processed_texts if t]) if processed_texts else 0,
            "filtered_out": len([t for t in processed_texts if not t])
        }
        
        return stats
