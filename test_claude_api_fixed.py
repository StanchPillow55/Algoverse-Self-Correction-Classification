#!/usr/bin/env python3
"""
Test Claude API functionality with proper environment variable loading.
"""

import os
import json
import time
from anthropic import Anthropic
from dotenv import load_dotenv

def test_claude_api():
    """Test Claude API with simple questions."""
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Check if API key is available
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("✗ Anthropic API key not found in .env file")
        return False
    
    print(f"✓ Anthropic API key loaded (ends with: ...{api_key[-6:] if api_key else 'None'})")
    
    # Initialize Anthropic client with the API key
    try:
        client = Anthropic(api_key=api_key)
        print("✓ Anthropic client initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize Anthropic client: {e}")
        return False
    
    # Test questions
    test_questions = [
        {"id": 1, "question": "What is 10 + 5?", "expected": "15"},
        {"id": 2, "question": "What is 8 × 3?", "expected": "24"},
        {"id": 3, "question": "What is 20 ÷ 4?", "expected": "5"},
        {"id": 4, "question": "What is the capital of France?", "expected": "Paris"},
        {"id": 5, "question": "What color do you get when you mix red and yellow?", "expected": "orange"}
    ]
    
    print(f"\n🧪 Testing Claude API with {len(test_questions)} questions...")
    print("=" * 60)
    
    successful_calls = 0
    results = []
    
    # Models to test
    models_to_test = [
        "claude-3-5-sonnet-20241022",  # Updated model ID
        "claude-3-haiku-20240307"
    ]
    
    for model in models_to_test:
        print(f"\n🤖 Testing model: {model}")
        print("-" * 40)
        
        model_successful = 0
        model_results = []
        
        for i, test in enumerate(test_questions, 1):
            print(f"\nQuestion {test['id']}: {test['question']}")
            
            try:
                # Make API call
                response = client.messages.create(
                    model=model,
                    max_tokens=100,
                    messages=[
                        {"role": "user", "content": test['question']}
                    ]
                )
                
                answer = response.content[0].text.strip()
                
                # Check if response contains expected answer
                contains_expected = test['expected'].lower() in answer.lower()
                
                print(f"✓ API Response: {answer}")
                print(f"Expected: {test['expected']} | Contains Expected: {'✓' if contains_expected else '✗'}")
                
                model_successful += 1
                successful_calls += 1
                model_results.append({
                    "question_id": test['id'],
                    "question": test['question'],
                    "expected": test['expected'],
                    "response": answer,
                    "contains_expected": contains_expected,
                    "success": True
                })
                
                # Small delay between calls
                time.sleep(0.5)
                
            except Exception as e:
                print(f"✗ API Error: {e}")
                model_results.append({
                    "question_id": test['id'],
                    "question": test['question'],
                    "expected": test['expected'],
                    "response": str(e),
                    "contains_expected": False,
                    "success": False
                })
        
        results.append({
            "model": model,
            "successful_calls": model_successful,
            "total_questions": len(test_questions),
            "results": model_results
        })
        
        print(f"\n📊 Model {model} Summary: {model_successful}/{len(test_questions)} successful")
        
        # If this model works, we can break (no need to test all models)
        if model_successful > 0:
            print(f"✅ Found working model: {model}")
            break
    
    # Overall summary
    print("\n" + "=" * 60)
    print(f"🔍 OVERALL TEST SUMMARY:")
    print(f"Total successful API calls: {successful_calls}")
    
    if successful_calls > 0:
        print("✅ Claude API is FUNCTIONAL")
        api_functional = True
    else:
        print("❌ Claude API has ISSUES")
        api_functional = False
    
    # Save results
    results_file = "claude_api_test_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "api_functional": api_functional,
            "total_successful_calls": successful_calls,
            "model_results": results
        }, f, indent=2)
    
    print(f"📄 Detailed results saved to: {results_file}")
    
    return api_functional

if __name__ == "__main__":
    print("🔧 Claude API Functionality Test")
    print("=" * 60)
    
    success = test_claude_api()
    exit(0 if success else 1)