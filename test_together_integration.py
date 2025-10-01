#!/usr/bin/env python3
"""
Test script to verify Together AI integration with Llama models
"""

import os
import json
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_together_api():
    """Test direct Together AI API call."""
    print("🔍 Testing Together AI API...")
    
    # Check for API key
    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        print("❌ TOGETHER_API_KEY not found in environment")
        print("Please add to .env file: TOGETHER_API_KEY=your_key_here")
        return False
    
    print(f"✅ API key found: {api_key[:8]}...")
    
    try:
        from together import Together
        
        client = Together(api_key=api_key)
        
        # Test with a simple prompt
        response = client.chat.completions.create(
            model="meta-llama/Llama-2-7b-chat-hf",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is 2+2? Answer with just the number."}
            ],
            temperature=0.0,
            max_tokens=10
        )
        
        answer = response.choices[0].message.content
        print(f"✅ API call successful! Response: {answer}")
        
        # Show usage if available
        if hasattr(response, 'usage'):
            print(f"📊 Token usage: {response.usage}")
        
        return True
        
    except Exception as e:
        print(f"❌ API call failed: {e}")
        return False

def test_learner_bot():
    """Test LearnerBot with Together AI provider."""
    print("\n🤖 Testing LearnerBot with Together AI...")
    
    try:
        from src.agents.learner import LearnerBot
        
        # Create a LearnerBot with Together AI provider
        bot = LearnerBot(provider="together", model="llama-7b")
        
        print(f"✅ LearnerBot created with model: {bot.model}")
        
        # Test answering a simple question
        question = "What is 15 + 27? Show your work."
        answer, confidence, full_response = bot.answer(question, [], None)
        
        print(f"📝 Question: {question}")
        print(f"✅ Answer: {answer[:100]}...")
        print(f"💯 Confidence: {confidence}")
        
        return True
        
    except Exception as e:
        print(f"❌ LearnerBot test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_config():
    """Test model configuration loading."""
    print("\n📋 Testing model configuration...")
    
    config_path = "configs/scaling_models.json"
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        return False
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Check Llama models
    llama_models = [m for m in config['models'] if 'llama' in m['name']]
    
    print(f"✅ Found {len(llama_models)} Llama models:")
    for model in llama_models:
        status = "✅" if model['provider'] == 'together' else "⚠️"
        print(f"  {status} {model['name']}: provider={model['provider']}, cost=${model['estimated_cost_per_1k_tokens']}/1k tokens")
    
    # Verify all are using Together AI
    all_together = all(m['provider'] == 'together' for m in llama_models)
    if all_together:
        print("✅ All Llama models configured to use Together AI")
    else:
        print("⚠️ Some Llama models not using Together AI")
    
    return all_together

def estimate_costs():
    """Estimate costs for Llama experiments."""
    print("\n💰 Cost Estimation for Llama experiments...")
    
    # Load config
    with open("configs/scaling_models.json", 'r') as f:
        config = json.load(f)
    
    llama_models = [m for m in config['models'] if 'llama' in m['name']]
    
    # Estimate for 100 samples, 3 turns each, ~2k tokens per turn
    samples = 100
    turns = 3
    tokens_per_turn = 2000
    
    print(f"📊 Assumptions: {samples} samples, {turns} turns, {tokens_per_turn} tokens/turn")
    
    total_cost = 0
    for model in llama_models:
        model_tokens = samples * turns * tokens_per_turn
        model_cost = (model_tokens / 1000) * model['estimated_cost_per_1k_tokens']
        total_cost += model_cost
        print(f"  {model['name']}: {model_tokens:,} tokens = ${model_cost:.2f}")
    
    print(f"💵 Total estimated cost for all Llama models: ${total_cost:.2f}")
    
    return True

def main():
    """Run all tests."""
    print("🚀 Together AI Integration Test Suite")
    print("=" * 50)
    
    tests = [
        ("API Connection", test_together_api),
        ("LearnerBot Integration", test_learner_bot),
        ("Model Configuration", test_model_config),
        ("Cost Estimation", estimate_costs)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n▶️ Running: {test_name}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ Test crashed: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status}: {test_name}")
    
    all_passed = all(success for _, success in results)
    if all_passed:
        print("\n🎉 All tests passed! Together AI integration is ready.")
        print("\n📝 Next steps:")
        print("1. Add TOGETHER_API_KEY to your .env file")
        print("2. Run: python run_toolqa_experiments.py --models llama-7b --samples 10")
        print("3. Monitor for proper execution and cost tracking")
    else:
        print("\n⚠️ Some tests failed. Please fix the issues above.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())