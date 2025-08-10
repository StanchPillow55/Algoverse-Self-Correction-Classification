"""
RTS Policy Simulator for Multi-Pass Self-Correction Traces
"""
import json
import random
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

@dataclass
class RTSDecision:
    """RTS policy decision output"""
    reprompt: bool
    prompt_id: Optional[str]
    confidence: float

class RTSPolicySimulator:
    """
    Simulates RTS policy decisions for multi-pass self-correction.
    
    This is a placeholder implementation that will be replaced with
    actual learned policy once we have training data.
    """
    
    def __init__(self, rts_templates_path: str = "rts_templates.json", max_turns: int = 5):
        self.max_turns = max_turns
        self.templates = self._load_templates(rts_templates_path)
        
    def _load_templates(self, path: str) -> Dict[str, Dict]:
        """Load RTS templates from JSON file"""
        try:
            with open(path, 'r') as f:
                templates_list = json.load(f)
            return {t['id']: t for t in templates_list}
        except FileNotFoundError:
            # Fallback minimal template set
            return {
                "neutral_verification": {
                    "id": "neutral_verification",
                    "style": "neutral", 
                    "cognitive_load": "medium",
                    "text": "Can you verify the accuracy of <ANSWER>?"
                }
            }
    
    def make_decision(self, 
                     error_label: str,
                     confidence: float, 
                     history: List[Dict],
                     turn_number: int) -> RTSDecision:
        """
        Make RTS policy decision based on error, confidence, and history.
        
        Args:
            error_label: Predicted error type (anchored, overcorrected, etc.)
            confidence: Model confidence in current answer (0-1)
            history: List of previous turns
            turn_number: Current turn number
            
        Returns:
            RTSDecision with reprompt decision and chosen prompt
        """
        # Simple heuristic policy (to be replaced with learned policy)
        
        # Don't exceed max turns
        if turn_number >= self.max_turns:
            return RTSDecision(reprompt=False, prompt_id=None, confidence=confidence)
        
        # High confidence + no error = stop
        if confidence > 0.8 and error_label in ["none", "unchanged_correct"]:
            return RTSDecision(reprompt=False, prompt_id=None, confidence=confidence)
            
        # Very low confidence = try supportive prompt
        if confidence < 0.3:
            prompt_id = self._select_prompt_by_style("supportive", history)
            return RTSDecision(reprompt=True, prompt_id=prompt_id, confidence=confidence)
            
        # Anchoring detected = challenge with adversarial prompt
        if error_label == "anchored" and confidence > 0.6:
            prompt_id = self._select_prompt_by_style("adversarial", history)
            return RTSDecision(reprompt=True, prompt_id=prompt_id, confidence=confidence)
            
        # Over-correction tendency = use calming prompt
        if error_label == "overcorrected":
            prompt_id = self._select_prompt_by_style("supportive", history) 
            return RTSDecision(reprompt=True, prompt_id=prompt_id, confidence=confidence)
            
        # Default: neutral verification
        prompt_id = self._select_prompt_by_style("neutral", history)
        return RTSDecision(reprompt=True, prompt_id=prompt_id, confidence=confidence)
    
    def _select_prompt_by_style(self, style: str, history: List[Dict]) -> str:
        """Select prompt by style, avoiding recent repeats"""
        # Get prompts matching style
        candidates = [t for t in self.templates.values() if t['style'] == style]
        
        if not candidates:
            # Fallback to any available prompt
            candidates = list(self.templates.values())
            
        # Avoid recently used prompts
        recent_prompt_ids = {turn.get('prompt_id') for turn in history[-2:]}
        unused_candidates = [c for c in candidates if c['id'] not in recent_prompt_ids]
        
        if unused_candidates:
            candidates = unused_candidates
            
        # Random selection from candidates
        return random.choice(candidates)['id']
    
    def simulate_error_and_confidence(self, answer: str, turn_number: int) -> Tuple[str, float]:
        """
        Simulate error detection and confidence estimation.
        
        This is a placeholder that generates realistic distributions.
        In the real system, this would use the trained classifier.
        """
        # Simulate confidence - tends to decrease with more turns (fatigue effect)
        base_confidence = random.uniform(0.2, 0.9)
        fatigue_factor = max(0.1, 1.0 - (turn_number * 0.1))
        confidence = base_confidence * fatigue_factor
        
        # Simulate error labels with realistic distribution
        error_options = ["none", "anchored", "overcorrected", "unchanged_correct"]
        
        if confidence > 0.7:
            # High confidence - more likely to be correct or anchored
            error_label = random.choices(
                ["none", "anchored", "unchanged_correct"], 
                weights=[0.6, 0.3, 0.1]
            )[0]
        elif confidence < 0.4:
            # Low confidence - more likely to have issues
            error_label = random.choices(
                ["overcorrected", "anchored", "none"], 
                weights=[0.5, 0.3, 0.2]
            )[0]
        else:
            # Medium confidence - balanced distribution
            error_label = random.choice(error_options)
            
        return error_label, confidence

def generate_multi_pass_trace(question: str, 
                             reference_answer: str,
                             initial_answer: str,
                             max_turns: int = 5) -> Dict:
    """
    Generate a complete multi-pass self-correction trace.
    
    Args:
        question: The input question
        reference_answer: Ground truth answer for evaluation
        initial_answer: Starting answer (turn 0)
        max_turns: Maximum number of correction turns
        
    Returns:
        Complete trace in specified JSON format
    """
    rts_policy = RTSPolicySimulator(max_turns=max_turns)
    
    # Initialize trace
    trace = {
        "question": question,
        "reference_answer": reference_answer,
        "turns": []
    }
    
    current_answer = initial_answer
    turn_number = 0
    
    while turn_number < max_turns:
        # Simulate error detection and confidence
        error_label, confidence = rts_policy.simulate_error_and_confidence(
            current_answer, turn_number
        )
        
        # Make RTS decision
        decision = rts_policy.make_decision(
            error_label=error_label,
            confidence=confidence,
            history=trace["turns"],
            turn_number=turn_number
        )
        
        # Record current turn
        turn_data = {
            "turn": turn_number,
            "answer": current_answer,
            "error": error_label,
            "confidence": round(confidence, 3),
            "prompt_id": decision.prompt_id if decision.reprompt else "none"
        }
        trace["turns"].append(turn_data)
        
        # Check if we should stop
        if not decision.reprompt:
            break
            
        # Generate next answer (simulated)
        current_answer = _simulate_answer_revision(
            current_answer, decision.prompt_id, rts_policy.templates
        )
        turn_number += 1
    
    return trace

def _simulate_answer_revision(current_answer: str, prompt_id: str, templates: Dict) -> str:
    """
    Simulate how an answer might change given a reprompt.
    
    This is a placeholder - in real usage, this would query the actual LLM.
    """
    if prompt_id == "none" or not prompt_id:
        return current_answer
        
    # Simple simulation: add variation based on prompt style
    template = templates.get(prompt_id, {})
    style = template.get("style", "neutral")
    
    modifications = {
        "supportive": ["(refined)", "(with more detail)", "(double-checked)"],
        "adversarial": ["(reconsidered)", "(with caveats)", "(alternative view)"],
        "concise": ["(simplified)", "(core answer)"],
        "exhaustive": ["(detailed analysis)", "(comprehensive view)"],
        "neutral": ["(verified)", "(clarified)"],
        "critique-revise": ["(revised)", "(corrected)"]
    }
    
    mod_options = modifications.get(style, ["(updated)"])
    modification = random.choice(mod_options)
    
    return f"{current_answer} {modification}"

# Example usage
if __name__ == "__main__":
    # Generate example trace
    trace = generate_multi_pass_trace(
        question="What is the capital of France?",
        reference_answer="Paris",
        initial_answer="Paris",
        max_turns=3
    )
    
    print(json.dumps(trace, indent=2))
