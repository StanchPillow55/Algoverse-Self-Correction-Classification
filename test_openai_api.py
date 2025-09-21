#!/usr/bin/env python3
"""
Quick test script to verify OpenAI API functionality with 5 basic math questions.
"""

import os
import openai
from openai import OpenAI
import json
import time
from dotenv import load_dotenv

def test_openai_api():
    """Test OpenAI API with 5 simple math questions."""
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Check if API key is available
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("✗ OpenAI API key not found in .env file")
        return False
    
    print(f"✓ OpenAI API key loaded (ends with: ...{api_key[-6:] if api_key else 'None'})")
    
    # Initialize OpenAI client
    try:
        client = OpenAI(api_key=api_key)
        print("✓ OpenAI client initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize OpenAI client: {e}")
        return False
    
    # Test questions
    test_questions = [
        {"id": 1, "question": "What is 15 + 27?", "expected": "42"},
        {"id": 2, "question": "What is 6 × 7?", "expected": "42"},
        {"id": 3, "question": "What is 100 ÷ 4?", "expected": "25"},
        {"id": 4, "question": "What is 2³ (2 to the power of 3)?", "expected": "8"},
        {"id": 5, "question": "If a pizza has 8 slices and you eat 3, how many are left?", "expected": "5"}
    ]
    
    print(f"\n🧪 Testing OpenAI API with {len(test_questions)} math questions...")
    print("=" * 60)
    
    successful_calls = 0
    results = []
    
    for i, test in enumerate(test_questions, 1):
        print(f"\nQuestion {test['id']}: {test['question']}")
        
        try:
            # Make API call
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful math assistant. Give concise, direct answers to math problems."},
                    {"role": "user", "content": test['question']}
                ],
                max_tokens=100,
                temperature=0.1
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Check if response contains expected answer
            contains_expected = test['expected'] in answer
            
            print(f"✓ API Response: {answer}")
            print(f"Expected: {test['expected']} | Contains Expected: {'✓' if contains_expected else '✗'}")
            
            successful_calls += 1
            results.append({
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
            results.append({
                "question_id": test['id'],
                "question": test['question'],
                "expected": test['expected'],
                "response": str(e),
                "contains_expected": False,
                "success": False
            })
    
    # Summary
    print("\n" + "=" * 60)
    print(f"🔍 TEST SUMMARY:")
    print(f"Successful API calls: {successful_calls}/{len(test_questions)}")
    
    correct_answers = sum(1 for r in results if r['success'] and r['contains_expected'])
    print(f"Correct answers: {correct_answers}/{len(test_questions)}")
    
    if successful_calls == len(test_questions):
        print("✅ OpenAI API is FUNCTIONAL")
        api_functional = True
    else:
        print("❌ OpenAI API has ISSUES")
        api_functional = False
    
    # Save results
    results_file = "openai_api_test_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_questions": len(test_questions),
            "successful_calls": successful_calls,
            "correct_answers": correct_answers,
            "api_functional": api_functional,
            "results": results
        }, f, indent=2)
    
    print(f"📄 Detailed results saved to: {results_file}")
    
    return api_functional

if __name__ == "__main__":
    print("🔧 OpenAI API Functionality Test")
    print("=" * 60)
    
    success = test_openai_api()
    exit(0 if success else 1)
