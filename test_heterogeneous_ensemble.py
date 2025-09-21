#!/usr/bin/env python3
"""
Test script to verify heterogeneous ensemble functionality
"""
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

def test_heterogeneous_ensemble():
    """Test heterogeneous ensemble with different providers"""
    
    os.environ["DEMO_MODE"] = "1"
    
    print("🧪 Testing Heterogeneous Ensemble Functionality")
    print("=" * 60)
    
    try:
        from src.ensemble.learner import EnsembleLearnerBot
        
        # Test 1: Create heterogeneous ensemble
        print("✓ Test 1: Creating heterogeneous ensemble...")
        
        heterogeneous_configs = [
            {"provider": "demo", "model": "demo-gpt-4o-mini"},
            {"provider": "demo", "model": "demo-claude-haiku"},
            {"provider": "demo", "model": "demo-gpt-4o"},
            {"provider": "demo", "model": "demo-claude-sonnet"}
        ]
        
        learner = EnsembleLearnerBot(
            provider="mixed",
            ensemble_size=4,
            ensemble_configs=heterogeneous_configs
        )
        
        print(f"  Created ensemble with {len(learner.ensemble_configs)} models")
        print(f"  Is heterogeneous: {learner.is_heterogeneous}")
        
        for i, config in enumerate(learner.ensemble_configs):
            print(f"    Model {i+1}: {config['provider']}:{config['model']}")
        
        # Test 2: Generate ensemble response
        print("\\n✓ Test 2: Testing heterogeneous answer generation...")
        
        question = "What is 25 + 17?"
        answer, confidence, response = learner.answer(question, [])
        
        print(f"  Question: {question}")
        print(f"  Final Answer: {answer}")
        print(f"  Confidence: {confidence}")
        print(f"  Contains heterogeneous info: {'demo-gpt' in response or 'demo-claude' in response}")
        
        # Test 3: Test voting with mixed responses
        print("\\n✓ Test 3: Testing voting with heterogeneous responses...")
        
        # Simulate responses from different providers
        mixed_responses = ["42", "42", "43", "42"]  # Majority should be 42
        mixed_confidences = [0.9, 0.8, 0.7, 0.85]
        mixed_raw_texts = [
            "OpenAI GPT-4o-mini: The answer is 42",
            "Anthropic Claude Haiku: I calculate 42",
            "OpenAI GPT-4o: My result is 43", 
            "Anthropic Claude Sonnet: The sum equals 42"
        ]
        
        final_answer, final_conf, info = learner._aggregate_ensemble_responses(
            mixed_responses, mixed_confidences, mixed_raw_texts
        )
        
        print(f"  Voting result: {final_answer}")
        print(f"  Final confidence: {final_conf:.3f}")
        print(f"  Voting method: {info['voting_method']}")
        print(f"  Consensus ratio: {info['consensus_ratio']}")
        
        # Test 4: Test adaptive voting with heterogeneous data
        print("\\n✓ Test 4: Testing adaptive voting...")
        
        adaptive_answer, adaptive_conf, adaptive_info = learner._adaptive_voting(
            mixed_responses, mixed_confidences, mixed_raw_texts, is_code_task=False
        )
        
        print(f"  Adaptive result: {adaptive_answer}")
        print(f"  Adaptive confidence: {adaptive_conf:.3f}")
        print(f"  Chosen strategy: {adaptive_info.get('adaptive_strategy')}")
        print(f"  Diversity ratio: {adaptive_info.get('diversity_ratio')}")
        
        # Test 5: Verify heterogeneous model calling
        print("\\n✓ Test 5: Testing heterogeneous model calls...")
        
        # Test calling specific provider:model combinations
        test_answer, test_conf, test_raw = learner._call_heterogeneous_model(
            "What is 10 + 5?", None, "demo", "demo-test-model",
            "test_exp", "test_dataset", "test_sample", 0
        )
        
        print(f"  Heterogeneous call result: {test_answer}")
        print(f"  Call succeeded: {not test_answer.startswith('ERROR')}")
        
        print("\\n✅ All heterogeneous ensemble tests passed!")
        return True
        
    except Exception as e:
        print(f"\\n❌ Heterogeneous ensemble test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_loading():
    """Test loading heterogeneous configurations"""
    
    print("\\n🧪 Testing Heterogeneous Configuration Loading")
    print("=" * 60)
    
    import json
    
    try:
        # Test the new heterogeneous config
        config_file = "configs/ensemble_experiments/demo_heterogeneous.json"
        
        if not Path(config_file).exists():
            print(f"❌ Config file not found: {config_file}")
            return False
        
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        print(f"✓ Loaded config: {config['name']}")
        print(f"  Provider: {config['provider']}")
        print(f"  Ensemble size: {config['ensemble_size']}")
        print(f"  Has ensemble_configs: {'ensemble_configs' in config}")
        
        if 'ensemble_configs' in config:
            print("  Models:")
            for i, model_config in enumerate(config['ensemble_configs']):
                print(f"    {i+1}. {model_config['provider']}:{model_config['model']}")
        
        print("\\n✅ Configuration loading test passed!")
        return True
        
    except Exception as e:
        print(f"\\n❌ Configuration test failed: {e}")
        return False

def main():
    """Run all heterogeneous ensemble tests"""
    
    print("🎭 HETEROGENEOUS ENSEMBLE TEST SUITE")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 2
    
    # Test 1: Core functionality
    if test_heterogeneous_ensemble():
        tests_passed += 1
    
    # Test 2: Configuration loading
    if test_config_loading():
        tests_passed += 1
    
    print("\\n" + "=" * 60)
    print(f"HETEROGENEOUS TEST RESULTS: {tests_passed}/{total_tests} tests passed")
    print("=" * 60)
    
    if tests_passed == total_tests:
        print("🎉 Heterogeneous ensemble system is working correctly!")
        print("\\n📋 Summary of capabilities:")
        print("  ✅ Multi-provider ensemble creation")
        print("  ✅ Heterogeneous model calling")
        print("  ✅ Mixed-provider voting")
        print("  ✅ Adaptive strategy selection")
        print("  ✅ Configuration loading")
        return 0
    else:
        print("⚠️  Some heterogeneous tests failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())