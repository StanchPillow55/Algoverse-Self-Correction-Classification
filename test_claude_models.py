#!/usr/bin/env python3
"""
Test different Claude model IDs to find which ones are available.
"""

import os
import json
from anthropic import Anthropic

# Initialize Anthropic client
client = Anthropic()

# Model IDs to test
claude_models_to_test = [
    "claude-3-5-sonnet-20241210",  # Current model ID we're using
    "claude-3-5-sonnet-20240620",
    "claude-3-5-sonnet-20241210", 
    "claude-3-haiku-20240307",  # Known working model
]

def test_model(model_id):
    """Test if a model ID works by making a simple API call."""
    try:
        response = client.messages.create(
            model=model_id,
            max_tokens=10,
            messages=[{"role": "user", "content": "Hello"}]
        )
        return True, response.content[0].text if response.content else "Success"
    except Exception as e:
        return False, str(e)

def main():
    print("Testing Claude model IDs...")
    print("=" * 50)
    
    working_models = []
    failed_models = []
    
    for model_id in claude_models_to_test:
        print(f"Testing {model_id}...")
        success, result = test_model(model_id)
        
        if success:
            print(f"  ✅ WORKS: {result}")
            working_models.append(model_id)
        else:
            print(f"  ❌ FAILED: {result}")
            failed_models.append((model_id, result))
        print()
    
    print("=" * 50)
    print("SUMMARY:")
    print(f"Working models ({len(working_models)}):")
    for model in working_models:
        print(f"  ✅ {model}")
    
    print(f"\nFailed models ({len(failed_models)}):")
    for model, error in failed_models:
        print(f"  ❌ {model}: {error}")
    
    # Save results to file
    results = {
        "working_models": working_models,
        "failed_models": [{"model": model, "error": error} for model, error in failed_models],
        "test_timestamp": "2025-01-16"
    }
    
    with open("claude_model_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to claude_model_test_results.json")

if __name__ == "__main__":
    main()