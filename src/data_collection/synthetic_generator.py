"""
Synthetic data generator for creating LLM error examples.

Uses various LLM APIs to generate examples of different error types
for training the classification model.
"""

import openai
from typing import List, Dict, Any, Optional
import pandas as pd
from ..utils.error_types import ErrorType, get_error_definition
from ..utils.config import Config
import logging

logger = logging.getLogger(__name__)


class SyntheticDataGenerator:
    """Generate synthetic examples of LLM errors for training data."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the synthetic data generator."""
        Config.load()
        self.api_key = api_key or Config.OPENAI_API_KEY
        if self.api_key:
            openai.api_key = self.api_key
        
    def generate_error_example(self, error_type: ErrorType, prompt: str) -> Dict[str, Any]:
        """Generate a single example of a specific error type."""
        error_def = get_error_definition(error_type)
        
        system_prompt = f"""
        You are helping create training data for an LLM error classification system.
        Generate a response that exhibits {error_def.name} characteristics.
        
        Description: {error_def.description}
        Key symptoms to include: {', '.join(error_def.symptoms)}
        
        Make the error subtle but detectable. The response should seem plausible
        but clearly exhibit the specified error type.
        """
        
        try:
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.8
            )
            
            generated_text = response.choices[0].message.content
            
            return {
                "prompt": prompt,
                "response": generated_text,
                "error_type": error_type.value,
                "error_severity": error_def.severity,
                "correction_strategy": error_def.correction_strategy,
                "symptoms_present": error_def.symptoms
            }
            
        except Exception as e:
            logger.error(f"Error generating example for {error_type}: {e}")
            return None
    
    def generate_dataset(self, 
                        prompts: List[str], 
                        samples_per_error: int = 10) -> pd.DataFrame:
        """Generate a complete dataset with examples of all error types."""
        data = []
        
        error_types = [et for et in ErrorType if et != ErrorType.NO_ERROR]
        
        for error_type in error_types:
            logger.info(f"Generating examples for {error_type.value}")
            
            for i, prompt in enumerate(prompts[:samples_per_error]):
                example = self.generate_error_example(error_type, prompt)
                if example:
                    example['sample_id'] = f"{error_type.value}_{i}"
                    data.append(example)
        
        # Generate some "no error" examples
        for i, prompt in enumerate(prompts[:samples_per_error]):
            no_error_example = self.generate_clean_example(prompt)
            if no_error_example:
                no_error_example['sample_id'] = f"no_error_{i}"
                data.append(no_error_example)
        
        return pd.DataFrame(data)
    
    def generate_clean_example(self, prompt: str) -> Dict[str, Any]:
        """Generate a clean, error-free response example."""
        system_prompt = """
        You are a helpful, accurate AI assistant. Provide a clear, concise,
        and well-reasoned response to the user's question. Avoid any signs of
        uncertainty, bias, overthinking, or other response issues.
        """
        
        try:
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            generated_text = response.choices[0].message.content
            
            return {
                "prompt": prompt,
                "response": generated_text,
                "error_type": ErrorType.NO_ERROR.value,
                "error_severity": "none",
                "correction_strategy": "none",
                "symptoms_present": []
            }
            
        except Exception as e:
            logger.error(f"Error generating clean example: {e}")
            return None
    
    def get_default_prompts(self) -> List[str]:
        """Get a set of default prompts for data generation."""
        return [
            "What is the capital of France?",
            "Explain quantum computing in simple terms.",
            "How do you make a basic pasta sauce?",
            "What are the main causes of climate change?",
            "Describe the process of photosynthesis.",
            "What is the difference between AI and machine learning?",
            "How do vaccines work?",
            "Explain the concept of compound interest.",
            "What are the benefits of regular exercise?",
            "How does the internet work?",
            "What is the significance of the Renaissance period?",
            "Explain the water cycle.",
            "How do you solve a quadratic equation?",
            "What are the main principles of democracy?",
            "Describe the structure of DNA."
        ]
