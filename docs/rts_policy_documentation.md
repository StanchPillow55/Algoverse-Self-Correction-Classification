# RTS Policy Head Documentation

## Overview

The RTS (Reprompt Template Selection) Policy Head implements a **contextual bandit** approach to intelligently decide:
1. **Whether to reprompt** (binary decision)
2. **Which prompt to use** (from available template set)

Based on the current context (error type, confidence, turn history), the policy learns to maximize reward over time.

## Problem Formulation

### Context
```python
Context = {
    detected_error: str,     # "anchored", "overcorrected", etc.
    confidence: float,       # Model confidence [0, 1]
    last_prompt_id: str,     # Previous prompt used (optional)
    turn_index: int         # Current turn number
}
```

### Actions
```python
Actions = {
    "none",                  # No reprompt
    "p_try_again",          # Simple retry
    "p_think_step_by_step", # Encourage systematic thinking
    "p_think_less",         # Reduce overthinking
    "p_are_you_sure",       # Challenge confidence
    "p_explain_why",        # Request justification
    # ... additional prompts from RTS template set
}
```

### Reward Function
```python
Reward = delta_accuracy - λ * token_cost
```
- **delta_accuracy**: Change in correctness (-1, 0, +1)
- **λ**: Cost penalty coefficient (default: 0.001)
- **token_cost**: Number of tokens used in reprompt

## Algorithms Implemented

### 1. Thompson Sampling

**Principle**: Bayesian approach using Beta distributions to model action value uncertainty.

**Algorithm Outline**:
```python
# For each context-action pair, maintain:
alpha[context][action] = success_count + 1  # Prior
beta[context][action] = failure_count + 1   # Prior

def select_action(context, actions):
    for action in actions:
        sample[action] = Beta(alpha[context][action], beta[context][action])
    return argmax(sample)

def update(context, action, reward):
    if reward > 0:
        alpha[context][action] += reward
    else:
        beta[context][action] += abs(reward)
```

**Advantages**:
- Natural exploration-exploitation trade-off
- Principled uncertainty quantification
- Good theoretical properties

### 2. ε-greedy with Decaying Epsilon

**Principle**: Balance exploration (random action) and exploitation (best known action).

**Algorithm Outline**:
```python
def select_action(context, actions, step):
    epsilon = initial_epsilon * (decay_rate ** step)
    
    if random() < epsilon:
        return random_choice(actions)  # Explore
    else:
        return argmax(q_values[context])  # Exploit

def update(context, action, reward):
    count[context][action] += 1
    q_values[context][action] += (reward - q_values[context][action]) / count[context][action]
```

**Advantages**:
- Simple and intuitive
- Guaranteed exploration
- Decaying epsilon balances exploration/exploitation over time

## Implementation Architecture

### Core Classes

#### 1. `RTSContext` (Data Structure)
```python
@dataclass
class RTSContext:
    detected_error: str
    confidence: float
    last_prompt_id: Optional[str]
    turn_index: int
```

#### 2. `RTSAction` (Data Structure)
```python
@dataclass
class RTSAction:
    reprompt: bool
    prompt_id: str
    expected_reward: float = 0.0
```

#### 3. `BanditAlgorithm` (Abstract Base)
```python
class BanditAlgorithm(ABC):
    @abstractmethod
    def select_action(self, context_key: str, actions: List[str]) -> str:
        pass
    
    @abstractmethod
    def update(self, context_key: str, action: str, reward: float):
        pass
```

#### 4. `RTSPolicyEnhanced` (Main Interface)
```python
class RTSPolicyEnhanced:
    def select_prompt(self, context: RTSContext) -> RTSAction:
        """Main interface for action selection"""
        
    def update_policy(self, context: RTSContext, action: RTSAction, 
                     delta_accuracy: int, token_cost: int = 0):
        """Update policy with observed outcome"""
```

## Context Discretization

To handle continuous context variables, the policy discretizes contexts into discrete keys:

### Confidence Binning
```python
if confidence < 0.4:    bin = "low"
elif confidence < 0.7:  bin = "medium"  
else:                   bin = "high"
```

### Turn Index Binning
```python
if turn_index <= 1:     bin = "early"
elif turn_index <= 3:   bin = "mid"
else:                   bin = "late"
```

### Final Context Key
```python
context_key = f"{detected_error}_{confidence_bin}_{turn_bin}"
# Example: "anchored_high_early"
```

## Usage Examples

### Basic Usage
```python
from rts_policy_enhanced import RTSPolicyEnhanced, RTSContext

# Initialize policy
policy = RTSPolicyEnhanced(
    algorithm="thompson_sampling",
    lambda_token_cost=0.001
)

# Create context
context = RTSContext(
    detected_error="anchored",
    confidence=0.8,
    last_prompt_id=None,
    turn_index=1
)

# Select action
action = policy.select_prompt(context)
print(f"Reprompt: {action.reprompt}, Prompt: {action.prompt_id}")

# Simulate outcome and update
delta_accuracy = 1  # Improvement
token_cost = 50     # Tokens used
policy.update_policy(context, action, delta_accuracy, token_cost)
```

### Integration with Error Classifier
```python
from error_confidence_classifier import ErrorConfidenceTrainer
from rts_policy_enhanced import RTSPolicyEnhanced, RTSContext

# Initialize components
classifier = ErrorConfidenceTrainer(config)
classifier.load_model("model.pt", "vectorizer.pkl")
policy = RTSPolicyEnhanced(algorithm="thompson_sampling")

# Multi-turn correction loop
def multi_turn_correction(question, initial_answer, max_turns=5):
    current_answer = initial_answer
    
    for turn in range(max_turns):
        # Classify current state
        prediction = classifier.predict(
            initial_answer=initial_answer,
            revised_answer=current_answer,
            reprompt_id="none"
        )
        
        # Create context
        context = RTSContext(
            detected_error=prediction['failure_mode'],
            confidence=prediction['confidence_score'],
            last_prompt_id=None,
            turn_index=turn
        )
        
        # Select action
        action = policy.select_prompt(context)
        
        if not action.reprompt:
            break
            
        # Apply reprompt and get new answer
        # (This would involve actual LLM call)
        new_answer = apply_reprompt(current_answer, action.prompt_id)
        
        # Calculate reward and update policy
        delta_acc = evaluate_improvement(current_answer, new_answer, reference)
        policy.update_policy(context, action, delta_acc, estimate_token_cost(action.prompt_id))
        
        current_answer = new_answer
    
    return current_answer
```

## Performance Analysis

### Reward Calculation Examples

| Scenario | Delta Acc | Token Cost | Reward (λ=0.001) |
|----------|-----------|------------|-------------------|
| Perfect fix, short prompt | +1 | 20 | 0.98 |
| No change, short prompt | 0 | 20 | -0.02 |
| Makes worse, long prompt | -1 | 100 | -1.10 |
| Improvement, expensive | +1 | 500 | 0.50 |

### Context Space Analysis

With current discretization:
- **Error types**: 6 (anchored, overcorrected, etc.)
- **Confidence bins**: 3 (low, medium, high)  
- **Turn bins**: 3 (early, mid, late)
- **Total contexts**: 6 × 3 × 3 = 54 discrete states

This manageable state space allows effective learning with reasonable sample efficiency.

### Sample Efficiency

**Thompson Sampling**: Typically requires 10-20 samples per context-action to converge to good policies.

**ε-greedy**: Requires more samples but is more robust to non-stationary environments.

## Configuration Options

### Algorithm Selection
```python
# Thompson Sampling (recommended)
policy = RTSPolicyEnhanced(algorithm="thompson_sampling")

# ε-greedy with custom parameters
policy = RTSPolicyEnhanced(
    algorithm="epsilon_greedy",
    # Algorithm will use EpsilonGreedy(epsilon=0.1, decay_rate=0.99)
)
```

### Cost Sensitivity
```python
# Low cost sensitivity (more willing to reprompt)
policy = RTSPolicyEnhanced(lambda_token_cost=0.0001)

# High cost sensitivity (conservative reprompting)
policy = RTSPolicyEnhanced(lambda_token_cost=0.01)
```

### Template Integration
```python
# Use custom template file
policy = RTSPolicyEnhanced(rts_templates_path="custom_templates.json")
```

## Monitoring and Analysis

### Action Statistics
```python
stats = policy.get_action_statistics()
print(f"Algorithm: {stats['algorithm']}")
print(f"Total actions: {stats['total_actions']}")
print(f"Templates loaded: {stats['templates_loaded']}")

# Algorithm-specific stats
if stats['algorithm'] == 'thompson_sampling':
    print(f"Parameters learned: {stats['alpha_params']}")
elif stats['algorithm'] == 'epsilon_greedy':
    print(f"Current epsilon: {stats['current_epsilon']:.3f}")
```

### Policy Persistence
```python
# Save trained policy
policy.save_policy("trained_rts_policy.pkl")

# Load for continued learning/inference
new_policy = RTSPolicyEnhanced(algorithm="thompson_sampling")
new_policy.load_policy("trained_rts_policy.pkl")
```

## Best Practices

### 1. Context Design
- **Keep discrete**: Use meaningful bins for continuous variables
- **Balance granularity**: Too fine → slow learning, too coarse → poor performance
- **Include relevant features**: Turn history and error types are crucial

### 2. Reward Engineering
- **Scale appropriately**: Ensure token costs don't dominate accuracy gains
- **Consider delayed rewards**: Multi-turn improvements may require longer observation windows
- **Handle edge cases**: Very short/long responses, edge model behaviors

### 3. Exploration Management
- **Thompson Sampling**: Good default choice, handles exploration naturally
- **ε-greedy**: Better for non-stationary environments, easier to tune
- **Warm-up period**: Consider using random policy initially to gather data

### 4. Integration Patterns
```python
# Good: Clear separation of concerns
error_prediction = classifier.predict(...)
context = create_context(error_prediction, turn_info)
action = policy.select_prompt(context)

# Good: Proper reward feedback loop
outcome = execute_action(action)
reward = calculate_reward(outcome)
policy.update_policy(context, action, outcome.delta_acc, outcome.tokens)
```

## Extensions and Future Work

### 1. Hierarchical Policies
- Separate policies for "should reprompt?" vs "which prompt?"
- Meta-learning across different domains/tasks

### 2. Neural Contextual Bandits
- Replace discrete context keys with learned representations
- Use neural networks for value estimation

### 3. Multi-Armed Bandit Ensembles
- Combine multiple bandit algorithms
- Dynamic algorithm selection based on context

### 4. Transfer Learning
- Pre-train policies on simulated data
- Adapt policies across different model types

This RTS Policy Head provides a solid foundation for intelligent prompt selection in multi-turn self-correction scenarios, with proven algorithms and practical engineering considerations.
