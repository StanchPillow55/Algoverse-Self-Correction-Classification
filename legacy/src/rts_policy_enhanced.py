"""
Enhanced RTS Policy Head with Multiple Algorithms

This module provides both Thompson Sampling and ε-greedy algorithms
for contextual bandit prompt selection, with full integration with
the RTS template system.
"""

import numpy as np
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
import pickle
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

@dataclass
class RTSContext:
    """Structured context for RTS policy decisions"""
    detected_error: str
    confidence: float
    last_prompt_id: Optional[str]
    turn_index: int
    
    def to_dict(self) -> Dict:
        return {
            'detected_error': self.detected_error,
            'confidence': self.confidence,
            'last_prompt_id': self.last_prompt_id,
            'turn_index': self.turn_index
        }

@dataclass
class RTSAction:
    """Action taken by the RTS policy"""
    reprompt: bool
    prompt_id: str
    expected_reward: float = 0.0

class BanditAlgorithm(ABC):
    """Abstract base class for bandit algorithms"""
    
    @abstractmethod
    def select_action(self, context_key: str, actions: List[str]) -> str:
        pass
    
    @abstractmethod
    def update(self, context_key: str, action: str, reward: float):
        pass

class ThompsonSampling(BanditAlgorithm):
    """Thompson Sampling algorithm using Beta distributions"""
    
    def __init__(self):
        self.alpha = {}  # Success counts
        self.beta = {}   # Failure counts
        
    def select_action(self, context_key: str, actions: List[str]) -> str:
        samples = {}
        
        for action in actions:
            key = f"{context_key}_{action}"
            # Get alpha (successes) and beta (failures) for this context-action
            alpha = self.alpha.get(key, 1.0)  # Prior
            beta = self.beta.get(key, 1.0)    # Prior
            
            # Sample from Beta distribution
            sample = np.random.beta(alpha, beta)
            samples[action] = sample
            
        # Return action with highest sample
        return max(samples, key=samples.get)
    
    def update(self, context_key: str, action: str, reward: float):
        key = f"{context_key}_{action}"
        
        # Convert reward to success/failure
        # Reward > 0 is success, reward <= 0 is failure
        if reward > 0:
            self.alpha[key] = self.alpha.get(key, 1.0) + reward
        else:
            self.beta[key] = self.beta.get(key, 1.0) + abs(reward)

class EpsilonGreedy(BanditAlgorithm):
    """ε-greedy algorithm with decaying epsilon"""
    
    def __init__(self, epsilon: float = 0.1, decay_rate: float = 0.99):
        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.q_values = {}  # Q-values for each context-action
        self.counts = {}    # Action counts
        self.step = 0
        
    def select_action(self, context_key: str, actions: List[str]) -> str:
        self.step += 1
        current_epsilon = self.epsilon * (self.decay_rate ** self.step)
        
        if np.random.random() < current_epsilon:
            # Explore: random action
            return np.random.choice(actions)
        else:
            # Exploit: best known action
            q_vals = {}
            for action in actions:
                key = f"{context_key}_{action}"
                q_vals[action] = self.q_values.get(key, 0.0)
            
            return max(q_vals, key=q_vals.get)
    
    def update(self, context_key: str, action: str, reward: float):
        key = f"{context_key}_{action}"
        
        # Update count
        self.counts[key] = self.counts.get(key, 0) + 1
        count = self.counts[key]
        
        # Update Q-value with incremental mean
        old_q = self.q_values.get(key, 0.0)
        self.q_values[key] = old_q + (1 / count) * (reward - old_q)

class RTSPolicyEnhanced:
    """
    Enhanced RTS Policy Head with configurable bandit algorithms.
    
    Features:
    - Thompson Sampling or ε-greedy algorithms
    - Integration with RTS template system
    - Context discretization strategies
    - Reward calculation with token cost
    - Policy state persistence
    """
    
    def __init__(self,
                 algorithm: str = "thompson_sampling",  # or "epsilon_greedy"
                 lambda_token_cost: float = 0.001,
                 rts_templates_path: str = "rts_templates.json"):
        
        # Load RTS templates
        self.templates = self._load_templates(rts_templates_path)
        self.action_prompt_ids = list(self.templates.keys()) + ["none"]
        
        # Initialize bandit algorithm
        if algorithm == "thompson_sampling":
            self.bandit = ThompsonSampling()
        elif algorithm == "epsilon_greedy":
            self.bandit = EpsilonGreedy()
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
            
        self.algorithm_name = algorithm
        self.lambda_token_cost = lambda_token_cost
        
        logger.info(f"RTS Policy initialized with {algorithm}, {len(self.action_prompt_ids)} actions")
    
    def _load_templates(self, path: str) -> Dict[str, Dict]:
        """Load RTS templates from JSON file"""
        try:
            with open(path, 'r') as f:
                templates_list = json.load(f)
            return {t['id']: t for t in templates_list}
        except FileNotFoundError:
            logger.warning(f"RTS templates not found at {path}. Using default actions.")
            return {
                "p_try_again": {"text": "Please try again."},
                "p_think_step_by_step": {"text": "Let's think step by step."},
                "p_are_you_sure": {"text": "Are you sure about <ANSWER>?"}
            }
    
    def _discretize_context(self, context: RTSContext) -> str:
        """Convert context to discrete key for bandit algorithm"""
        
        # Discretize confidence
        if context.confidence < 0.4:
            conf_bin = "low"
        elif context.confidence < 0.7:
            conf_bin = "medium"
        else:
            conf_bin = "high"
        
        # Discretize turn index  
        if context.turn_index <= 1:
            turn_bin = "early"
        elif context.turn_index <= 3:
            turn_bin = "mid"
        else:
            turn_bin = "late"
            
        return f"{context.detected_error}_{conf_bin}_{turn_bin}"
    
    def select_prompt(self, context: RTSContext) -> RTSAction:
        """
        Main interface: Select reprompt action given context.
        
        Args:
            context: RTSContext with detected_error, confidence, etc.
            
        Returns:
            RTSAction with reprompt decision and prompt_id
        """
        context_key = self._discretize_context(context)
        
        # Use bandit algorithm to select action
        selected_prompt_id = self.bandit.select_action(context_key, self.action_prompt_ids)
        
        # Determine reprompt decision
        reprompt = selected_prompt_id != "none"
        
        # Create action
        action = RTSAction(
            reprompt=reprompt,
            prompt_id=selected_prompt_id
        )
        
        logger.info(f"Context: {context_key} → Action: {selected_prompt_id} (reprompt: {reprompt})")
        return action
    
    def update_policy(self, 
                      context: RTSContext,
                      action: RTSAction, 
                      delta_accuracy: int,
                      token_cost: int = 0):
        """
        Update policy with observed outcome.
        
        Pseudocode for Update:
        1. Calculate reward = delta_accuracy - λ * token_cost
        2. Get context key from discretization
        3. Update bandit algorithm with (context, action, reward)
        4. Log the update for monitoring
        """
        # Calculate reward
        reward = self._calculate_reward(delta_accuracy, token_cost)
        
        # Get context key
        context_key = self._discretize_context(context)
        
        # Update bandit algorithm
        self.bandit.update(context_key, action.prompt_id, reward)
        
        logger.info(f"Policy updated: context={context_key}, action={action.prompt_id}, "
                   f"delta_acc={delta_accuracy}, tokens={token_cost}, reward={reward:.3f}")
    
    def _calculate_reward(self, delta_accuracy: int, token_cost: int) -> float:
        """Calculate reward = delta_accuracy - λ * token_cost"""
        return float(delta_accuracy) - (self.lambda_token_cost * token_cost)
    
    def get_action_statistics(self) -> Dict:
        """Get statistics about action selection and performance"""
        stats = {
            "algorithm": self.algorithm_name,
            "total_actions": len(self.action_prompt_ids),
            "templates_loaded": len(self.templates)
        }
        
        if isinstance(self.bandit, ThompsonSampling):
            stats["alpha_params"] = len(self.bandit.alpha)
            stats["beta_params"] = len(self.bandit.beta)
        elif isinstance(self.bandit, EpsilonGreedy):
            stats["q_values"] = len(self.bandit.q_values)
            stats["total_steps"] = self.bandit.step
            stats["current_epsilon"] = self.bandit.epsilon * (self.bandit.decay_rate ** self.bandit.step)
            
        return stats
    
    def save_policy(self, path: str = "rts_policy_enhanced.pkl"):
        """Save the complete policy state"""
        policy_data = {
            "algorithm_name": self.algorithm_name,
            "lambda_token_cost": self.lambda_token_cost,
            "bandit_state": self.bandit.__dict__,
            "action_prompt_ids": self.action_prompt_ids
        }
        
        with open(path, 'wb') as f:
            pickle.dump(policy_data, f)
        logger.info(f"Policy saved to {path}")
    
    def load_policy(self, path: str = "rts_policy_enhanced.pkl"):
        """Load a saved policy state"""
        try:
            with open(path, 'rb') as f:
                policy_data = pickle.load(f)
            
            # Restore bandit state
            for key, value in policy_data["bandit_state"].items():
                setattr(self.bandit, key, value)
                
            logger.info(f"Policy loaded from {path}")
        except FileNotFoundError:
            logger.warning(f"No policy file found at {path}. Starting fresh.")

# Demonstration and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test both algorithms
    for algorithm in ["thompson_sampling", "epsilon_greedy"]:
        print(f"\n{'='*50}")
        print(f"Testing {algorithm.upper()}")
        print(f"{'='*50}")
        
        policy = RTSPolicyEnhanced(algorithm=algorithm)
        
        # Simulate some interactions
        contexts = [
            RTSContext("anchored", 0.8, None, 1),
            RTSContext("overcorrected", 0.3, "p_try_again", 2),
            RTSContext("unchanged_correct", 0.9, None, 1)
        ]
        
        for i, context in enumerate(contexts):
            print(f"\nScenario {i+1}:")
            print(f"Context: {context.to_dict()}")
            
            # Select action
            action = policy.select_prompt(context)
            print(f"Selected: reprompt={action.reprompt}, prompt_id={action.prompt_id}")
            
            # Simulate outcome and update
            delta_acc = np.random.choice([-1, 0, 1])  # Random outcome
            token_cost = 50 if action.reprompt else 0
            
            policy.update_policy(context, action, delta_acc, token_cost)
            print(f"Updated with: delta_acc={delta_acc}, cost={token_cost}")
        
        print(f"\nFinal statistics: {policy.get_action_statistics()}")
