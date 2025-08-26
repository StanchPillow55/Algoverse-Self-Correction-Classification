"""
RTS Policy Head - Contextual Bandit for Prompt Selection

This module implements a contextual bandit policy using Thompson Sampling
to select the optimal reprompt action based on the current context.

Context = {detected_error, confidence, last_prompt_id, turn_index}
Actions = {none, p_try_again, p_think_step_by_step, ...}
Reward = delta_accuracy - λ * token_cost
"""

import numpy as np
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
import pickle

logger = logging.getLogger(__name__)

@dataclass
class Action:
    """Represents a single action the policy can take"""
    prompt_id: str  # 'none' for no reprompt

@dataclass
class PolicyState:
    """Represents the learned state of the policy"""
    action_value_estimates: Dict[str, Dict[str, float]]
    action_counts: Dict[str, Dict[str, int]]
    
    def __init__(self, actions: List[Action], context_keys: List[str]):
        self.action_value_estimates = {key: {action.prompt_id: 0.0 for action in actions} 
                                     for key in context_keys}
        self.action_counts = {key: {action.prompt_id: 0 for action in actions} 
                              for key in context_keys}

class RTSPolicyHead:
    """
    Contextual bandit policy head using Thompson Sampling.
    
    Algorithm Outline (Thompson Sampling):
    1. Maintain prior distributions for each action's reward (Beta distribution).
    2. For each context, sample from the posterior of each action.
    3. Select the action with the highest sample.
    4. Observe reward and update the posterior of the chosen action.
    """
    
    def __init__(self, 
                 actions: Optional[List[Action]] = None,
                 lambda_token_cost: float = 0.001,
                 exploration_factor: float = 0.1):
        self.actions = actions or [
            Action("none"),
            Action("p_try_again"),
            Action("p_think_step_by_step"),
            Action("p_think_less"),
            Action("p_are_you_sure"),
            Action("p_explain_why")
        ]
        self.lambda_token_cost = lambda_token_cost
        self.exploration_factor = exploration_factor  # For ε-greedy fallback
        
        # Context keys for state representation
        self.context_keys = ['error_mode', 'confidence_bin']
        self.state = self._initialize_state()
        
    def _initialize_state(self) -> PolicyState:
        """Initialize the policy state with default context values"""
        context_keys = []
        # Binned confidence levels
        for conf in ['low', 'medium', 'high']:
            # Failure modes
            for error in ['anchored', 'overcorrected', 'corrected', 'unchanged_correct', 'none']:
                context_keys.append(f"{error}_{conf}")
        return PolicyState(self.actions, context_keys)

    def _get_context_key(self, context: Dict) -> str:
        """Create a context key for state representation"""
        error_mode = context.get('detected_error', 'none')
        confidence = context.get('confidence', 0.5)
        
        # Discretize confidence into bins
        if confidence < 0.4:
            conf_bin = 'low'
        elif confidence < 0.7:
            conf_bin = 'medium'
        else:
            conf_bin = 'high'
        
        return f"{error_mode}_{conf_bin}"

    def select_prompt(self, context: Dict) -> Tuple[bool, str]:
        """
        Select the best reprompt action using Thompson Sampling.
        
        Args:
            context: {detected_error, confidence, last_prompt_id, turn_index}
        
        Returns:
            Tuple of (reprompt_bool, prompt_id)
        """
        context_key = self._get_context_key(context)
        
        # Thompson Sampling: sample from Beta distribution for each action
        # (successes, failures) -> (alpha, beta)
        # For simplicity, we use value estimates as proxy for Beta params
        
        sampled_values = {}
        for action in self.actions:
            action_id = action.prompt_id
            
            # Beta distribution parameters from historical performance
            # successes = value_estimate * counts, failures = (1-value_estimate) * counts
            # We simplify this to a normal distribution for easier sampling
            
            value_estimate = self.state.action_value_estimates[context_key][action_id]
            count = self.state.action_counts[context_key][action_id]
            
            # Add exploration for actions with few counts
            if count < 5:
                # Encourage exploration of less-tried actions
                sampled_value = np.random.normal(loc=value_estimate, scale=1.0)
            else:
                # Exploit known good actions
                sampled_value = np.random.normal(
                    loc=value_estimate, 
                    scale=1.0 / np.sqrt(count + 1)
                )
                
            sampled_values[action_id] = sampled_value
        
        # Epsilon-greedy exploration for robustness
        if np.random.random() < self.exploration_factor:
            # Choose a random action
            best_action_id = np.random.choice([a.prompt_id for a in self.actions])
        else:
            # Choose action with highest sampled value
            best_action_id = max(sampled_values, key=sampled_values.get)
        
        # Determine if we should reprompt
        reprompt_bool = best_action_id != 'none'
        
        logger.info(f"Context: {context_key}, Selected action: {best_action_id}, Reprompt: {reprompt_bool}")
        return reprompt_bool, best_action_id

    def update_policy(self, 
                      context: Dict, 
                      action_id: str,
                      reward: float):
        """
        Update the policy with observed reward.
        
        Pseudocode:
        1. Get context key from current state
        2. Update counts for (context, action) pair
        3. Update value estimate using incremental update rule:
           Q_new = Q_old + (1/N) * (reward - Q_old)
        """
        context_key = self._get_context_key(context)
        
        # Update counts
        self.state.action_counts[context_key][action_id] += 1
        count = self.state.action_counts[context_key][action_id]
        
        # Update value estimate
        old_value = self.state.action_value_estimates[context_key][action_id]
        # Incremental update rule (more numerically stable)
        new_value = old_value + (1 / count) * (reward - old_value)
        
        self.state.action_value_estimates[context_key][action_id] = new_value
        
        logger.info(f"Updated policy: context={context_key}, action={action_id}, reward={reward:.3f}, new_value={new_value:.3f}")

    def calculate_reward(self, 
                         delta_accuracy: int, 
                         token_cost: int = 0) -> float:
        """
        Calculate reward based on delta_accuracy and token cost.
        
        Reward = delta_accuracy - λ * token_cost
        """
        reward = float(delta_accuracy) - (self.lambda_token_cost * token_cost)
        return reward
    
    def save_policy_state(self, path: str = "rts_policy_state.pkl"):
        """
        Save the policy state (action values and counts) to a file.
        """
        with open(path, 'wb') as f:
            pickle.dump(self.state, f)
        logger.info(f"RTS policy state saved to {path}")
        
    def load_policy_state(self, path: str = "rts_policy_state.pkl"):
        """
        Load a saved policy state.
        """
        try:
            with open(path, 'rb') as f:
                self.state = pickle.load(f)
            logger.info(f"RTS policy state loaded from {path}")
        except FileNotFoundError:
            logger.warning(f"No policy state file found at {path}. Starting with a new policy.")

# Example usage
if __name__ == "__main__":
    # Setup
    logging.basicConfig(level=logging.INFO)
    
    rts_policy = RTSPolicyHead(
        lambda_token_cost=0.001, 
        exploration_factor=0.2
    )
    
    print("\n--- Initial Policy State ---")
    print(json.dumps(rts_policy.state.action_value_estimates, indent=2))
    
    # Simulate a scenario
    context = {
        'detected_error': 'anchored',
        'confidence': 0.8, 
        'last_prompt_id': 'p_try_again',
        'turn_index': 1
    }
    
    print("\n--- Selecting Action ---")
    reprompt, prompt_id = rts_policy.select_prompt(context)
    
    print(f"\nSelected prompt: {prompt_id}, Reprompt: {reprompt}")
    
    # Simulate observing a reward
    reward = rts_policy.calculate_reward(delta_accuracy=+1, token_cost=50)
    
    print(f"\n--- Updating Policy with Reward: {reward:.3f} ---")
    rts_policy.update_policy(context, prompt_id, reward)
    
    print("\n--- Updated Policy State ---")
    print(json.dumps(rts_policy.state.action_value_estimates, indent=2))
    
    # Save the policy state
    rts_policy.save_policy_state()
