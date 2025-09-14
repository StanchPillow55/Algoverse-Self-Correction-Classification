# Replicate Performance Improvements

## Overview

This document describes the performance improvements made to Replicate API calls in the self-correction classification system, specifically to address the significant slowdown when using Llama models compared to GPT-4.

## Problem

Llama models via Replicate API were significantly slower than GPT-4 due to:

1. **No rate limiting**: Direct `replicate.run()` calls without intelligent backoff
2. **No retry logic**: Failed requests weren't retried with exponential backoff
3. **No token-based throttling**: No consideration of token usage limits
4. **Generator overhead**: Inefficient collection of streaming responses

## Solution

### 1. Added `safe_replicate_run()` Wrapper

Created a symmetrical wrapper in `src/utils/rate_limit.py` that mirrors the existing `safe_openai_chat_completion()` and `safe_anthropic_messages_create()` functions.

**Key features:**
- Intelligent rate limiting with exponential backoff
- Token-based throttling
- Automatic retry logic for failed requests
- Consistent logging and error handling
- Decorrelated jitter to prevent thundering herd

### 2. Updated LearnerBot Implementation

Modified `src/agents/learner.py` to use the new wrapper:

```python
# Before
response = replicate.run(self.model, input={...})

# After  
response = safe_replicate_run(self.model, input_params={...})
```

## Expected Performance Improvements

### Speed Improvements
- **Reduced API failures**: Intelligent retry logic reduces failed requests
- **Better rate limiting**: Prevents hitting rate limits that cause delays
- **Optimized token estimation**: More accurate token counting for throttling

### Reliability Improvements
- **Exponential backoff**: Reduces server load and improves success rates
- **Consistent error handling**: Better logging and debugging
- **Token-aware throttling**: Prevents quota exhaustion

### Estimated Impact on Phase 3

For a typical GSM8K dataset (~1000 problems, 3 turns average = 3000 API calls):

- **Before**: ~2070 seconds for GPT-4, likely 6000-10000+ seconds for Llama-70B
- **After**: Expected 20-40% reduction in Llama-70B completion time
- **Reliability**: Fewer failed requests and retries needed

## Testing

### Test Scripts

1. **`test_replicate_wrapper.py`**: Basic functionality test
2. **`benchmark_replicate_performance.py`**: Performance comparison

### Running Tests

```bash
# Set your Replicate API token
export REPLICATE_API_TOKEN=your-token-here

# Test basic functionality
python test_replicate_wrapper.py

# Benchmark performance
python benchmark_replicate_performance.py
```

## Usage

The improvements are automatically applied when using:

```bash
# Phase 3 with Llama models
python scripts/run_scaling_simple.py --phase 3 --dataset data/gsm8k/gsm8k.jsonl

# Or any existing pipeline that uses Replicate
python -m src.main run --provider replicate --model meta/meta-llama-3-70b
```

## Configuration

The wrapper respects existing environment variables:

- `MAX_CONCURRENCY`: Maximum concurrent requests (default: 2)
- `RPS_LIMIT`: Requests per second limit (default: 2.0)
- `TPM_LIMIT`: Tokens per minute limit (default: 120000)
- `MAX_RETRIES`: Maximum retry attempts (default: 6)
- `RETRIES_ENABLED`: Enable/disable retries (default: 1)

## Monitoring

Rate limiting activity is logged to `logs/rate_limit.log` with entries like:
```
1234567890.123 pre_wait=1.50
1234567890.124 backoff=2.30 attempt=2
```

## Future Improvements

1. **Alternative Providers**: Consider Groq or Together AI for faster Llama inference
2. **Batch Processing**: Implement request batching for even better performance
3. **Model Selection**: Use smaller Llama models (7B, 13B) for faster inference
4. **Caching**: Implement response caching for repeated queries

## Files Modified

- `src/utils/rate_limit.py`: Added `safe_replicate_run()` function
- `src/agents/learner.py`: Updated `_call_replicate()` to use new wrapper
- `test_replicate_wrapper.py`: Test script (new)
- `benchmark_replicate_performance.py`: Benchmark script (new)
- `REPLICATE_PERFORMANCE_IMPROVEMENTS.md`: This documentation (new)
