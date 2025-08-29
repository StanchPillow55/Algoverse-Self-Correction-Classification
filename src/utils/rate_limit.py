"""
Rate limiting and retry logic for OpenAI API calls.

Provides exponential backoff with decorrelated jitter, token budgeting,
and concurrency control to prevent rate limit errors.
"""

import asyncio
import logging
import os
import random
import time
from collections import defaultdict, deque
from functools import wraps
from threading import Lock
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

# Simple backoff log (optional)
RATE_LIMIT_LOG = os.getenv("RATE_LIMIT_LOG", "logs/rate_limit.log")
_osmakedir = os.makedirs
try:
    _osmakedir(os.path.dirname(RATE_LIMIT_LOG), exist_ok=True)
except Exception:
    pass

def _rl_log(msg: str):
    try:
        path = os.getenv("RATE_LIMIT_LOG", RATE_LIMIT_LOG)
        parent = os.path.dirname(path)
        if parent:
            try:
                _osmakedir(parent, exist_ok=True)
            except Exception:
                pass
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(f"{time.time():.3f} {msg}\n")
    except Exception:
        pass

# Environment configuration
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "2"))
RPS_LIMIT = float(os.getenv("RPS_LIMIT", "2.0"))  # requests per second
TPM_LIMIT = int(os.getenv("TPM_LIMIT", "120000"))  # tokens per minute
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "6"))
RETRIES_ENABLED = os.getenv("RETRIES_ENABLED", "1") == "1"

# Global state for rate limiting
_global_semaphore = None
_request_times = deque()  # timestamps of recent requests
_token_usage = deque()    # (timestamp, token_count) pairs
_lock = Lock()


def _get_semaphore():
    """Get or create the global concurrency semaphore."""
    global _global_semaphore
    if _global_semaphore is None:
        _global_semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    return _global_semaphore


def _check_rate_limits(estimate_tokens: int = 1500) -> Optional[float]:
    """
    Check if we should delay before making a request.
    Returns delay in seconds, or None if no delay needed.
    """
    with _lock:
        now = time.time()
        
        # Clean old request timestamps (older than 1 second)
        while _request_times and now - _request_times[0] > 1.0:
            _request_times.popleft()
            
        # Clean old token usage (older than 60 seconds)
        while _token_usage and now - _token_usage[0][0] > 60.0:
            _token_usage.popleft()
        
        # Check RPS limit
        if len(_request_times) >= RPS_LIMIT:
            delay_for_rps = 1.0 - (now - _request_times[0])
            if delay_for_rps > 0:
                return delay_for_rps
        
        # Check TPM limit
        current_tokens = sum(tokens for _, tokens in _token_usage)
        if current_tokens + estimate_tokens > TPM_LIMIT:
            # Wait until the oldest token usage expires
            if _token_usage:
                delay_for_tpm = 60.0 - (now - _token_usage[0][0])
                if delay_for_tpm > 0:
                    return delay_for_tpm
        
        return None


def _record_request(tokens_used: int = 1500):
    """Record a successful request for rate limiting."""
    with _lock:
        now = time.time()
        _request_times.append(now)
        _token_usage.append((now, tokens_used))


def _parse_retry_after(exception: Exception) -> Optional[float]:
    """Extract Retry-After value from rate limit exception."""
    # Handle OpenAI SDK exceptions
    if hasattr(exception, 'response') and hasattr(exception.response, 'headers'):
        retry_after = exception.response.headers.get('Retry-After')
        if retry_after:
            try:
                return float(retry_after)
            except ValueError:
                pass
    
    # Fallback: check exception message for hints
    error_str = str(exception).lower()
    if 'retry after' in error_str:
        import re
        match = re.search(r'retry after (\d+(?:\.\d+)?)', error_str)
        if match:
            return float(match.group(1))
    
    return None


def call_with_backoff_sync(
    func: Callable,
    *args,
    estimate_tokens: int = 1500,
    **kwargs
) -> Any:
    """
    Execute a function with exponential backoff and rate limiting.
    
    Args:
        func: Function to call
        *args: Positional arguments for func
        estimate_tokens: Estimated token usage for this call
        **kwargs: Keyword arguments for func
        
    Returns:
        Result of func(*args, **kwargs)
        
    Raises:
        Exception: Final exception after all retries exhausted
    """
    if not RETRIES_ENABLED:
        return func(*args, **kwargs)
    
    last_exception = None
    base_delay = 1.0
    prev_delay = base_delay
    
    for attempt in range(MAX_RETRIES + 1):
        try:
            # Check rate limits before making request
            delay = _check_rate_limits(estimate_tokens)
            if delay and delay > 0:
                logger.warning(f"Rate limit pre-check: waiting {delay:.2f}s"); _rl_log(f"pre_wait={delay:.2f}")
                time.sleep(delay)
            
            # Make the request
            result = func(*args, **kwargs)
            
            # Record successful request
            _record_request(estimate_tokens)
            
            if attempt > 0:
                logger.info(f"Request succeeded after {attempt} retries")
            
            return result
            
        except Exception as e:
            last_exception = e
            
            # Check if this is a retriable error
            error_str = str(e).lower()
            is_rate_limit = any(term in error_str for term in [
                'rate limit', 'too many requests', '429', 'quota'
            ])
            is_server_error = any(term in error_str for term in [
                '500', '502', '503', '504', 'server error', 'timeout'
            ])
            
            if not (is_rate_limit or is_server_error):
                # Non-retriable error, re-raise immediately
                raise e
            
            if attempt >= MAX_RETRIES:
                # Final attempt failed
                logger.error(f"Request failed after {MAX_RETRIES} retries: {e}")
                raise e
            
            # Calculate backoff delay
            retry_after = _parse_retry_after(e)
            if retry_after:
                _rl_log(f"retry_after={retry_after:.2f}")
                delay = retry_after
            else:
                # Decorrelated jitter: delay = random(base_delay, prev_delay * 3)
                delay = random.uniform(base_delay, prev_delay * 3)
                delay = min(delay, 60.0)  # Cap at 60 seconds
            prev_delay = delay
            
            logger.warning(f"API error (attempt {attempt + 1}/{MAX_RETRIES + 1}): {e}. Retrying in {delay:.2f}s"); _rl_log(f"backoff={delay:.2f} attempt={attempt+1}")
            time.sleep(delay)
    
    # Should never reach here, but just in case
    raise last_exception


def estimate_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """
    Rough token estimation for different models.
    
    Args:
        text: Input text
        model: Model name
        
    Returns:
        Estimated token count
    """
    # Rough estimation: ~4 characters per token for most models
    base_tokens = len(text) // 4
    
    # Add overhead for chat completion format
    overhead = 50
    
    # Model-specific adjustments
    if "gpt-4" in model.lower():
        multiplier = 1.1  # GPT-4 is slightly more token-heavy
    else:
        multiplier = 1.0
    
    return int((base_tokens + overhead) * multiplier)


# Convenience function for common OpenAI chat completion pattern
def safe_openai_chat_completion(client, messages, model="gpt-4o-mini", **kwargs):
    """
    Wrapper for OpenAI chat completion with automatic rate limiting.
    
    Args:
        client: OpenAI client instance
        messages: List of chat messages
        model: Model name
        **kwargs: Additional arguments for chat completion
        
    Returns:
        OpenAI chat completion response
    """
    # Estimate tokens for all messages
    total_text = " ".join(msg.get("content", "") for msg in messages)
    tokens = estimate_tokens(total_text, model)
    
    def _make_request():
        return client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs
        )
    
    return call_with_backoff_sync(_make_request, estimate_tokens=tokens)


if __name__ == "__main__":
    # Simple test
    def test_func():
        print("Test function called")
        return "success"
    
    result = call_with_backoff_sync(test_func)
    print(f"Result: {result}")
