"""
Direct Error-to-Prompt Mapping Policy

This module implements the revised RTS policy that:
1. Takes errors above threshold with their probabilities
2. Maps each error directly to a specific prompt
3. Returns a list of prompts to be executed (concatenated or sequential)

Key Changes from Original:
- Input: errors_above_thresh = [(anchored, 0.7), (cognitive_overload, 0.32)]
- Action set: {anchored: p_are_you_sure, cognitive_overload: p_think_less, ...}
- Output: List of prompts to apply directly
"""

import json
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)

class ErrorType(Enum):
    """Enumeration of error types that can be detected"""
    ANCHORED = "anchored"
    OVERCORRECTED = "overcorrected" 
    CORRECTED = "corrected"
    UNCHANGED_CORRECT = "unchanged_correct"
    WAVERING = "wavering"
    PERFECTIONISM = "perfectionism"
    COGNITIVE_OVERLOAD = "cognitive_overload"
    PROMPT_BIAS = "prompt_bias"
    OVERTHINKING = "overthinking"

@dataclass
class ErrorDetection:
    """Represents a detected error with probability"""
    error_type: str
    probability: float
    
    def __post_init__(self):
        if self.probability < 0 or self.probability > 1:
            raise ValueError(f"Probability must be between 0 and 1, got {self.probability}")

@dataclass
class PromptAction:
    """Represents a prompt to be applied"""
    prompt_id: str
    prompt_text: str
    error_type: str
    priority: float  # Based on error probability
    
@dataclass
class PolicyOutput:
    """Output of the direct mapping policy"""
    should_reprompt: bool
    prompt_actions: List[PromptAction]
    execution_strategy: str  # "concatenated" or "sequential"
    reasoning: str

class DirectMappingPolicy:
    """
    Direct Error-to-Prompt Mapping Policy
    
    Maps detected errors directly to specific prompts without learning.
    This implements a rule-based approach where each error type has
    a predefined set of appropriate prompts.
    """
    
    def __init__(self, 
                 error_prompt_mapping: Optional[Dict[str, str]] = None,
                 rts_templates_path: str = "rts_templates.json",
                 threshold: float = 0.25,
                 max_prompts_per_turn: int = 3,
                 execution_strategy: str = "concatenated"):
        
        self.threshold = threshold
        self.max_prompts_per_turn = max_prompts_per_turn
        self.execution_strategy = execution_strategy
        
        # Load RTS templates
        self.rts_templates = self._load_rts_templates(rts_templates_path)
        
        # Default error-to-prompt mapping
        self.error_prompt_mapping = error_prompt_mapping or self._create_default_mapping()
        
        logger.info(f"Direct mapping policy initialized with {len(self.error_prompt_mapping)} error mappings")
        
    def _load_rts_templates(self, path: str) -> Dict[str, Dict]:
        """Load RTS templates from JSON file"""
        try:
            with open(path, 'r') as f:
                templates_list = json.load(f)
            return {t['id']: t for t in templates_list}
        except FileNotFoundError:
            logger.warning(f"RTS templates not found at {path}. Using minimal defaults.")
            return {
                "p_try_again": {"text": "Please try again.", "style": "supportive"},
                "p_are_you_sure": {"text": "Are you sure about <ANSWER>?", "style": "adversarial"},
                "p_think_step_by_step": {"text": "Let's think step by step.", "style": "neutral"},
                "p_think_less": {"text": "Keep it simple. What's your gut answer?", "style": "concise"}
            }
    
    def _create_default_mapping(self) -> Dict[str, str]:
        """
        Create default error-to-prompt mapping based on research insights.
        
        Mapping Logic:
        - anchored: Need adversarial challenge to break fixation
        - overcorrected: Need supportive/calming prompt to reduce anxiety
        - cognitive_overload: Need simplifying prompt to reduce complexity
        - perfectionism: Need concise prompt to prevent overthinking  
        - wavering: Need step-by-step structure
        - overthinking: Need simplifying prompt
        """
        return {
            ErrorType.ANCHORED.value: "adversarial_challenge",
            ErrorType.OVERCORRECTED.value: "calming_trust_instinct", 
            ErrorType.COGNITIVE_OVERLOAD.value: "focused_essentials",
            ErrorType.PERFECTIONISM.value: "concise_confirmation",
            ErrorType.WAVERING.value: "step_by_step_break_down",
            ErrorType.OVERTHINKING.value: "focused_core_issue",
            ErrorType.PROMPT_BIAS.value: "perspective_shift_outsider",
            
            # Fallbacks for other error types
            ErrorType.CORRECTED.value: "supportive_confidence_boost",
            ErrorType.UNCHANGED_CORRECT.value: "neutral_verification"
        }
    
    def select_prompts(self, 
                      errors_above_threshold: List[Tuple[str, float]],
                      context: Optional[Dict] = None) -> PolicyOutput:
        """
        Main interface: Select prompts based on detected errors.
        
        Args:
            errors_above_threshold: List of (error_type, probability) tuples
            context: Optional context information
            
        Returns:
            PolicyOutput with prompts to apply
        """
        # If no errors detected, don't reprompt
        if not errors_above_threshold:
            return PolicyOutput(
                should_reprompt=False,
                prompt_actions=[],
                execution_strategy="none",
                reasoning="No errors detected above threshold"
            )
        
        # Convert to ErrorDetection objects
        error_detections = [
            ErrorDetection(error_type, prob) 
            for error_type, prob in errors_above_threshold
        ]
        
        # Sort by probability (highest first)
        error_detections.sort(key=lambda x: x.probability, reverse=True)
        
        # Map errors to prompts
        prompt_actions = []
        used_prompts = set()  # Avoid duplicate prompts
        
        for error_detection in error_detections:
            if len(prompt_actions) >= self.max_prompts_per_turn:
                break
                
            # Get prompt ID for this error type
            prompt_id = self.error_prompt_mapping.get(error_detection.error_type)
            
            if prompt_id and prompt_id not in used_prompts:
                # Get prompt template
                template = self.rts_templates.get(prompt_id)
                
                if template:
                    prompt_action = PromptAction(
                        prompt_id=prompt_id,
                        prompt_text=template.get('text', f"Address {error_detection.error_type}"),
                        error_type=error_detection.error_type,
                        priority=error_detection.probability
                    )
                    prompt_actions.append(prompt_action)
                    used_prompts.add(prompt_id)
                else:
                    logger.warning(f"Template not found for prompt_id: {prompt_id}")
        
        # Generate reasoning
        error_summary = ", ".join([f"{ed.error_type} ({ed.probability:.2f})" for ed in error_detections])
        prompt_summary = ", ".join([pa.prompt_id for pa in prompt_actions])
        
        reasoning = (f"Detected errors: {error_summary}. "
                   f"Selected {len(prompt_actions)} prompts: {prompt_summary}")
        
        return PolicyOutput(
            should_reprompt=len(prompt_actions) > 0,
            prompt_actions=prompt_actions,
            execution_strategy=self.execution_strategy,
            reasoning=reasoning
        )
    
    def format_prompts(self, 
                      policy_output: PolicyOutput,
                      current_answer: str) -> List[str]:
        """
        Format prompts for execution, replacing placeholders.
        
        Args:
            policy_output: Output from select_prompts
            current_answer: Current answer to insert in prompts
            
        Returns:
            List of formatted prompt strings
        """
        if not policy_output.should_reprompt:
            return []
        
        formatted_prompts = []
        
        for prompt_action in policy_output.prompt_actions:
            prompt_text = prompt_action.prompt_text
            
            # Replace <ANSWER> placeholder
            if '<ANSWER>' in prompt_text:
                prompt_text = prompt_text.replace('<ANSWER>', current_answer)
            
            formatted_prompts.append(prompt_text)
        
        return formatted_prompts
    
    def combine_prompts(self, 
                       formatted_prompts: List[str],
                       strategy: str = "concatenated") -> str:
        """
        Combine multiple prompts according to execution strategy.
        
        Args:
            formatted_prompts: List of formatted prompt strings
            strategy: "concatenated" or "sequential"
            
        Returns:
            Combined prompt string
        """
        if not formatted_prompts:
            return ""
        
        if len(formatted_prompts) == 1:
            return formatted_prompts[0]
        
        if strategy == "concatenated":
            # Combine all prompts with separators
            return " | ".join(formatted_prompts)
        
        elif strategy == "sequential":
            # For now, just concatenate with numbers
            # In a full implementation, this would execute one at a time
            numbered_prompts = [f"{i+1}. {prompt}" for i, prompt in enumerate(formatted_prompts)]
            return "\n".join(numbered_prompts)
        
        else:
            logger.warning(f"Unknown strategy: {strategy}, using concatenated")
            return " | ".join(formatted_prompts)
    
    def get_mapping_stats(self) -> Dict:
        """Get statistics about the error-prompt mapping"""
        return {
            "total_error_types": len(self.error_prompt_mapping),
            "mapped_errors": list(self.error_prompt_mapping.keys()),
            "unique_prompts": len(set(self.error_prompt_mapping.values())),
            "execution_strategy": self.execution_strategy,
            "max_prompts_per_turn": self.max_prompts_per_turn,
            "threshold": self.threshold
        }
    
    def update_mapping(self, error_type: str, prompt_id: str):
        """Update the error-to-prompt mapping"""
        self.error_prompt_mapping[error_type] = prompt_id
        logger.info(f"Updated mapping: {error_type} -> {prompt_id}")
    
    def add_custom_template(self, prompt_id: str, template: Dict):
        """Add a custom prompt template"""
        self.rts_templates[prompt_id] = template
        logger.info(f"Added custom template: {prompt_id}")

# Integration class that combines classifier and policy
class IntegratedErrorCorrectionSystem:
    """
    Integrated system combining multi-label classifier and direct mapping policy.
    
    This represents the complete revised system:
    1. Multi-label classifier outputs error probabilities
    2. Policy maps errors above threshold to specific prompts
    3. Prompts are formatted and combined for execution
    """
    
    def __init__(self,
                 classifier,  # MultiLabelErrorTrainer instance
                 policy: DirectMappingPolicy,
                 threshold: float = 0.25):
        
        self.classifier = classifier
        self.policy = policy
        self.threshold = threshold
        
    def process_correction_step(self,
                               initial_answer: str,
                               revised_answer: str,
                               reprompt_id: str = "none") -> Dict:
        """
        Complete processing step: classify errors and select prompts.
        
        Args:
            initial_answer: Original answer
            revised_answer: Current answer
            reprompt_id: Previous reprompt used
            
        Returns:
            Dictionary with classification results and prompt recommendations
        """
        # Step 1: Classify errors
        classification_result = self.classifier.predict(
            initial_answer=initial_answer,
            revised_answer=revised_answer,
            reprompt_id=reprompt_id,
            threshold=self.threshold
        )
        
        # Step 2: Select prompts based on errors
        errors_above_threshold = classification_result['errors_above_threshold']
        policy_output = self.policy.select_prompts(errors_above_threshold)
        
        # Step 3: Format prompts if reprompting is needed
        formatted_prompts = []
        combined_prompt = ""
        
        if policy_output.should_reprompt:
            formatted_prompts = self.policy.format_prompts(policy_output, revised_answer)
            combined_prompt = self.policy.combine_prompts(
                formatted_prompts, 
                policy_output.execution_strategy
            )
        
        return {
            'classification': classification_result,
            'policy_decision': policy_output,
            'formatted_prompts': formatted_prompts,
            'combined_prompt': combined_prompt,
            'should_reprompt': policy_output.should_reprompt,
            'reasoning': policy_output.reasoning
        }

# Example usage and testing
def demonstrate_system():
    """Demonstrate the integrated error correction system"""
    
    print("🔬 Demonstrating Revised Error Correction System")
    print("=" * 60)
    
    # Initialize policy with default mappings
    policy = DirectMappingPolicy(
        threshold=0.25,
        max_prompts_per_turn=2,
        execution_strategy="concatenated"
    )
    
    # Example error detections from classifier
    test_cases = [
        {
            "name": "Single High-Confidence Error",
            "errors": [("anchored", 0.75)],
            "answer": "I think the answer is definitely X"
        },
        {
            "name": "Multiple Errors Above Threshold", 
            "errors": [("cognitive_overload", 0.45), ("perfectionism", 0.32)],
            "answer": "Well, it could be A, but maybe B is better, unless C..."
        },
        {
            "name": "Low Confidence Errors",
            "errors": [("anchored", 0.15), ("wavering", 0.08)],
            "answer": "The answer is Y"
        },
        {
            "name": "Mixed Error Types",
            "errors": [("anchored", 0.65), ("overthinking", 0.30), ("prompt_bias", 0.28)],
            "answer": "Based on my previous analysis, I believe..."
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 Test Case {i}: {test_case['name']}")
        print("-" * 40)
        
        # Apply policy
        policy_output = policy.select_prompts(test_case["errors"])
        
        print(f"Errors detected: {test_case['errors']}")
        print(f"Should reprompt: {policy_output.should_reprompt}")
        
        if policy_output.should_reprompt:
            print(f"Selected prompts: {len(policy_output.prompt_actions)}")
            
            # Format and combine prompts
            formatted_prompts = policy.format_prompts(policy_output, test_case["answer"])
            combined_prompt = policy.combine_prompts(formatted_prompts)
            
            print(f"Formatted prompts:")
            for j, prompt in enumerate(formatted_prompts, 1):
                print(f"  {j}. {prompt}")
            
            print(f"Combined prompt: {combined_prompt}")
        
        print(f"Reasoning: {policy_output.reasoning}")
    
    # Show mapping statistics
    print(f"\n📊 Policy Statistics:")
    stats = policy.get_mapping_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demonstrate_system()
