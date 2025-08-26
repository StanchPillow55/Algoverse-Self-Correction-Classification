"""
Synthetic data generator for creating LLM error examples.

Uses various LLM APIs to generate examples of different error types
for training the classification model.
"""

try:
    import openai
except ImportError:
    print("OpenAI library not installed. Using placeholder for API calls.")
    openai = None

from typing import List, Dict, Any, Optional
import pandas as pd
from ..utils.error_types import ErrorType, get_error_definition
from ..utils.config import Config
import logging
import random

logger = logging.getLogger(__name__)


class SyntheticDataGenerator:
    """Generate synthetic examples of LLM errors for training data."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the synthetic data generator with API placeholder support."""
        Config.load()
        self.api_key = api_key or Config.OPENAI_API_KEY
        self.use_api = False
        
        if self.api_key and openai:
            openai.api_key = self.api_key
            self.use_api = True
            logger.info("Using OpenAI API for data generation")
        else:
            logger.warning("API key not available. Using placeholder synthetic data generation.")
        
    def generate_error_example(self, error_type: ErrorType, prompt: str) -> Dict[str, Any]:
        """Generate a single example of a specific error type."""
        error_def = get_error_definition(error_type)
        
        if self.use_api and openai:
            return self._generate_api_example(error_type, prompt, error_def)
        else:
            return self._generate_placeholder_example(error_type, prompt, error_def)
    
    def _generate_api_example(self, error_type: ErrorType, prompt: str, error_def) -> Dict[str, Any]:
        """Generate example using OpenAI API."""
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
    
    def _generate_placeholder_example(self, error_type: ErrorType, prompt: str, error_def) -> Dict[str, Any]:
        """Generate placeholder example when API is not available."""
        # Create synthetic responses that exhibit the error characteristics
        placeholder_responses = {
            ErrorType.ANSWER_WAVERING: [
                "Well, it might be X, but actually it could also be Y. On second thought, maybe Z is correct. I'm not entirely sure though.",
                "The answer is probably A, although B seems reasonable too. Actually, let me reconsider - it might be C after all.",
                "I think the solution is this, but then again, that approach might not work. Perhaps we should try something else instead."
            ],
            ErrorType.PROMPT_BIAS: [
                "Based on what you've suggested in your question, the answer is clearly aligned with that perspective.",
                "As your question implies, this is definitely the case and there's no need to consider alternatives.",
                "Given the framing of your question, I completely agree with the underlying assumption."
            ],
            ErrorType.OVERTHINKING: [
                "This is a complex question that requires careful consideration of multiple factors, sub-factors, and meta-considerations...",
                "Let me break this down into seventeen different aspects, each with their own sub-components and interdependencies...",
                "While a simple answer might suffice, let's explore the deeper philosophical implications and theoretical frameworks..."
            ],
            ErrorType.COGNITIVE_OVERLOAD: [
                "So first we need to... wait, actually let me think about... no, that's not right. The steps are... hmm, I lost track.",
                "This involves multiple steps: A, then B, but wait, we need C first, or was it D? Let me start over...",
                "The process is straightforward: step 1, step 2, then we... actually, I think I mixed up the order somewhere."
            ],
            ErrorType.PERFECTIONISM_BIAS: [
                "While I could give you an answer, I should note that there are many caveats, exceptions, and potential edge cases to consider...",
                "The response is probably correct, though I must emphasize the numerous limitations and potential inaccuracies...",
                "This might be helpful, but please verify everything I say as I could be wrong on multiple levels..."
            ]
        }
        
        # Select a random response for the error type
        if error_type in placeholder_responses:
            selected_response = random.choice(placeholder_responses[error_type])
        else:
            selected_response = f"This is a placeholder response exhibiting {error_type.value} characteristics."
        
        return {
            "prompt": prompt,
            "response": selected_response,
            "error_type": error_type.value,
            "error_severity": error_def.severity,
            "correction_strategy": error_def.correction_strategy,
            "symptoms_present": error_def.symptoms
        }
    
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
        if self.use_api and openai:
            return self._generate_api_clean_example(prompt)
        else:
            return self._generate_placeholder_clean_example(prompt)
    
    def _generate_api_clean_example(self, prompt: str) -> Dict[str, Any]:
        """Generate clean example using OpenAI API."""
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
    
    def _generate_placeholder_clean_example(self, prompt: str) -> Dict[str, Any]:
        """Generate placeholder clean example when API is not available."""
        # Create clean responses based on common question patterns
        clean_responses = [
            "This is a straightforward answer to your question with clear, direct information.",
            "Here's a concise and accurate response that addresses your inquiry directly.",
            "The answer is clear and well-established, providing you with reliable information.",
            "This is a factual response that directly answers what you've asked.",
            "Here's the information you requested, presented clearly and accurately."
        ]
        
        selected_response = random.choice(clean_responses)
        
        return {
            "prompt": prompt,
            "response": selected_response,
            "error_type": ErrorType.NO_ERROR.value,
            "error_severity": "none",
            "correction_strategy": "none",
            "symptoms_present": []
        }
    
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
