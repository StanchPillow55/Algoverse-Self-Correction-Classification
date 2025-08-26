# Rate Limiting and Enhanced Feedback System

This document describes the rate limiting and enhanced evaluator feedback systems implemented to improve OpenAI API integration and provide actionable coaching for cognitive bias correction.

## Overview

The system implements three main enhancements:

1. **Rate Limiting**: Prevents OpenAI API rate limit errors with exponential backoff, token budgeting, and concurrency control
2. **Enhanced Feedback**: Converts bias labels to actionable coaching sentences
3. **Extended Trace Schema**: Adds rich evaluator feedback fields to trace logs

## Rate Limiting Implementation

### Core Features

The rate limiting system (`src/utils/rate_limit.py`) provides:

- **Exponential backoff with decorrelated jitter**: Avoids thundering herd problems
- **Retry-After header support**: Respects OpenAI's rate limit guidance
- **Token budgeting**: Tracks and limits tokens per minute (TPM)
- **Request rate limiting**: Controls requests per second (RPS)
- **Concurrency control**: Global semaphore limits concurrent requests

### Environment Configuration

Control the rate limiting behavior with these environment variables:

```bash
# Concurrency control (default: 2)
export MAX_CONCURRENCY=2

# Rate limits (default: 2 RPS, 120K TPM)
export RPS_LIMIT=2.0
export TPM_LIMIT=120000

# Retry behavior (default: 6 retries, enabled)
export MAX_RETRIES=6
export RETRIES_ENABLED=1

# Disable all retries for debugging
export RETRIES_ENABLED=0
```

### Usage

The rate limiting is automatically applied to OpenAI API calls in `src/agents/learner.py`:

```python
# Automatically wrapped with rate limiting
resp = safe_openai_chat_completion(
    client=client,
    model=self.model,
    messages=[...],
    temperature=0.2,
    max_tokens=40
)
```

## Enhanced Feedback System

### Bias-to-Coaching Mapping

The evaluator feedback system (`src/evaluator_feedback.py`) converts cognitive bias labels to actionable coaching:

| Bias Type | Coaching Feedback |
|-----------|-------------------|
| Confirmation | "You are hyper-confirming your training data instead of solving the problem. Pause and derive the answer from first principles, then recompute the key step." |
| Anchoring | "You are anchoring on numbers or phrases from the problem statement. Ignore the surface features and work through the logic step by step." |
| Fixation | "You are fixated on your initial approach and missing simpler solutions. Step back, consider alternative methods, and question your assumptions." |
| Overconfidence | "You are overconfident in your answer without proper verification. Double-check your work and consider where you might have made errors." |
| SunkCost | "You are persisting with a flawed approach because you've invested effort. Cut your losses and try a completely different strategy." |
| Availability | "You are defaulting to recent examples instead of the current problem. Focus on the specific details and requirements of this particular question." |
| Other | "Your reasoning shows systematic errors that need correction. Review your approach carefully and verify each logical step." |
| None | "Your reasoning is sound and your answer is correct. Good work applying logical thinking to solve the problem." |

### Usage

```python
from src.evaluator_feedback import coaching_from_bias

# Convert bias label to coaching
bias_label = "Confirmation"
coaching = coaching_from_bias(bias_label)
# Returns actionable coaching sentence
```

## Extended Trace Schema

### New Fields

The trace logging now includes enhanced evaluator feedback fields per turn:

```json
{
  "turn_index": 0,
  "prompt": "What is 2 + 2?",
  "response_text": "5",
  "is_correct": false,
  "evaluator_bias_label": "Confirmation",
  "evaluator_feedback": "You are hyper-confirming your training data instead of solving the problem. Pause and derive the answer from first principles, then recompute the key step.",
  "model_reported_confidence": 0.85,
  "turn_timestamp": "2025-08-19T09:58:49.133546Z"
}
```

### Backward Compatibility

- All existing fields are preserved
- New fields are added without breaking existing analysis scripts
- JSONL format remains one object per line

## Performance Monitoring

### Rate Limiting Logs

Monitor rate limiting behavior through log messages:

```
Rate limit pre-check: waiting 0.30s  # Proactive delay
API error (attempt 1/7): Rate limit exceeded. Retrying in 2.5s  # Retry with backoff
Request succeeded after 2 retries  # Success after retries
```

### Token and Request Tracking

The system tracks:
- Request timestamps (sliding 1-second window for RPS)
- Token usage (sliding 60-second window for TPM)
- Automatic cleanup of old tracking data

## Troubleshooting

### Persistent Rate Limit Errors

If you see repeated rate limit errors:

1. **Reduce concurrency**: Lower `MAX_CONCURRENCY` from 2 to 1
2. **Decrease RPS**: Lower `RPS_LIMIT` from 2.0 to 1.0
3. **Check token usage**: Monitor if `TPM_LIMIT` is too high for your tier

```bash
export MAX_CONCURRENCY=1
export RPS_LIMIT=1.0
export TPM_LIMIT=60000
```

### High Latency

If requests are slow due to excessive rate limiting:

1. **Check your OpenAI tier**: Higher tiers have higher limits
2. **Increase limits gradually**: Don't exceed your tier's limits
3. **Monitor actual usage**: Check OpenAI dashboard for usage patterns

### Debugging Rate Limiting

Disable rate limiting for debugging:

```bash
export RETRIES_ENABLED=0
```

This will make failures immediate and visible for troubleshooting.

## Performance Tuning Guidelines

### Optimal Settings by OpenAI Tier

**Tier 1 (New accounts)**:
```bash
export MAX_CONCURRENCY=1
export RPS_LIMIT=1.0
export TPM_LIMIT=30000
```

**Tier 2 (Usage-based)**:
```bash
export MAX_CONCURRENCY=2
export RPS_LIMIT=2.0
export TPM_LIMIT=120000
```

**Tier 3+ (Higher usage)**:
```bash
export MAX_CONCURRENCY=5
export RPS_LIMIT=5.0
export TPM_LIMIT=500000
```

### Monitoring Token Usage

Estimate token usage for different models:
- GPT-4o-mini: ~4 chars per token
- GPT-4: ~4 chars per token (slightly higher overhead)

### Concurrency vs. Latency Trade-offs

- **Higher concurrency**: Faster overall completion, higher risk of rate limits
- **Lower concurrency**: More reliable, slower overall completion
- **Recommended**: Start with concurrency=2, adjust based on error rates

## Examples

### Basic Usage

```python
# Set environment
export MAX_CONCURRENCY=2 RPS_LIMIT=2 TPM_LIMIT=120000 RETRIES_ENABLED=1

# Run experiment
python -m src.main run --dataset data/math_sample_20.csv --provider openai
```

### Conservative Settings

```python
# For unstable API or new accounts
export MAX_CONCURRENCY=1 RPS_LIMIT=1 TPM_LIMIT=30000 MAX_RETRIES=3

# Run experiment
python -m src.main run --dataset data/smoke_test_3.csv --provider openai
```

### Debug Mode

```python
# Disable retries to see raw errors
export RETRIES_ENABLED=0

# Run small test
python -m src.main run --dataset data/smoke_test_3.csv --provider openai
```

## Integration with Existing Analysis

The enhanced trace format is backward compatible with existing analysis scripts. New fields are simply additional and don't interfere with existing processing.

### Sample Analysis Code

```python
import json

# Load traces
with open('runs/latest/traces.jsonl', 'r') as f:
    for line in f:
        trace = json.loads(line)
        
        # Access new fields
        for turn in trace['turns']:
            if 'evaluator_bias_label' in turn:
                bias = turn['evaluator_bias_label']
                feedback = turn['evaluator_feedback']
                print(f"Bias: {bias}, Coaching: {feedback}")
```

## Testing

### Smoke Test

Verify the system with a small test:

```bash
# Create test dataset
head -4 data/math_sample_20.csv > data/smoke_test_3.csv

# Run smoke test
export MAX_CONCURRENCY=2 RPS_LIMIT=2 RETRIES_ENABLED=1
python -m src.main run --dataset data/smoke_test_3.csv --provider openai
```

Expected output:
- No unhandled RateLimitError exceptions
- Rate limiting log messages showing delays
- Non-zero accuracy (better than demo mode)
- Enhanced feedback in trace logs

### Validation

Verify trace format:

```python
import json

with open('runs/latest/traces.jsonl', 'r') as f:
    trace = json.loads(f.readline())
    
    # Check schema
    assert 'turns' in trace
    turn = trace['turns'][0]
    assert 'evaluator_bias_label' in turn
    assert 'evaluator_feedback' in turn
    assert len(turn['evaluator_feedback']) > 10
    
    print("✓ Enhanced trace schema validated")
```

This system provides robust, production-ready rate limiting with actionable feedback for cognitive bias correction while maintaining full backward compatibility.
