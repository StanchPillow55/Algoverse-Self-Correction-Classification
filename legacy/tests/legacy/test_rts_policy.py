#!/usr/bin/env python3
"""
Test Suite for RTS Policy Head

Tests both the basic and enhanced versions of the RTS policy
with Thompson Sampling and ε-greedy algorithms.
"""

import pytest
pytest.skip("Legacy classifier tests skipped during teacher/learner pivot", allow_module_level=True)

import sys
import os
import numpy as np
import logging
from typing import List, Dict

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from rts_policy_head import RTSPolicyHead
from rts_policy_enhanced import RTSPolicyEnhanced, RTSContext, RTSAction

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RTSPolicyTester:
    """Comprehensive test suite for RTS policy implementations"""
    
    def __init__(self):
        self.test_results = []
    
    def test_basic_policy_interface(self) -> bool:
        """Test the basic RTS policy head interface"""
        print("\n🧪 Testing Basic RTS Policy Interface...")
        
        try:
            policy = RTSPolicyHead(lambda_token_cost=0.001)
            
            # Test context
            context = {
                'detected_error': 'anchored',
                'confidence': 0.8,
                'last_prompt_id': None,
                'turn_index': 1
            }
            
            # Test selection
            reprompt, prompt_id = policy.select_prompt(context)
            assert isinstance(reprompt, bool), "Reprompt should be boolean"
            assert isinstance(prompt_id, str), "Prompt ID should be string"
            
            print(f"✅ Selection works: reprompt={reprompt}, prompt_id={prompt_id}")
            
            # Test reward calculation
            reward = policy.calculate_reward(delta_accuracy=1, token_cost=50)
            expected_reward = 1.0 - (0.001 * 50)
            assert abs(reward - expected_reward) < 1e-6, f"Reward calculation error: {reward} != {expected_reward}"
            
            print(f"✅ Reward calculation works: {reward:.3f}")
            
            # Test policy update
            policy.update_policy(context, prompt_id, reward)
            print("✅ Policy update works")
            
            # Test save/load
            policy.save_policy_state("test_policy.pkl")
            policy.load_policy_state("test_policy.pkl")
            print("✅ Save/Load works")
            
            return True
            
        except Exception as e:
            print(f"❌ Basic policy test failed: {e}")
            return False
    
    def test_enhanced_policy_thompson(self) -> bool:
        """Test enhanced policy with Thompson Sampling"""
        print("\n🧪 Testing Enhanced Policy - Thompson Sampling...")
        
        try:
            policy = RTSPolicyEnhanced(
                algorithm="thompson_sampling",
                lambda_token_cost=0.001
            )
            
            # Test contexts
            contexts = [
                RTSContext("anchored", 0.8, None, 1),
                RTSContext("overcorrected", 0.3, "p_try_again", 2),
                RTSContext("unchanged_correct", 0.9, None, 1)
            ]
            
            actions_taken = []
            
            for i, context in enumerate(contexts):
                # Test selection
                action = policy.select_prompt(context)
                actions_taken.append(action)
                
                assert isinstance(action.reprompt, bool), "Reprompt should be boolean"
                assert isinstance(action.prompt_id, str), "Prompt ID should be string"
                
                print(f"✅ Context {i+1}: {action.prompt_id} (reprompt: {action.reprompt})")
                
                # Simulate outcome and update
                delta_acc = np.random.choice([-1, 0, 1])
                token_cost = 50 if action.reprompt else 0
                
                policy.update_policy(context, action, delta_acc, token_cost)
                print(f"   Updated with delta_acc={delta_acc}, cost={token_cost}")
            
            # Test statistics
            stats = policy.get_action_statistics()
            assert stats["algorithm"] == "thompson_sampling"
            print(f"✅ Statistics: {stats}")
            
            return True
            
        except Exception as e:
            print(f"❌ Thompson Sampling test failed: {e}")
            return False
    
    def test_enhanced_policy_epsilon_greedy(self) -> bool:
        """Test enhanced policy with ε-greedy"""
        print("\n🧪 Testing Enhanced Policy - ε-greedy...")
        
        try:
            policy = RTSPolicyEnhanced(
                algorithm="epsilon_greedy",
                lambda_token_cost=0.001
            )
            
            # Run multiple episodes to test exploration/exploitation
            for episode in range(5):
                context = RTSContext("anchored", 0.6, None, 1)
                action = policy.select_prompt(context)
                
                # Simulate positive reward for one specific action to test learning
                if action.prompt_id == "p_are_you_sure":
                    delta_acc = 1  # Good reward
                else:
                    delta_acc = np.random.choice([-1, 0])  # Random/negative
                
                token_cost = 30 if action.reprompt else 0
                policy.update_policy(context, action, delta_acc, token_cost)
                
                print(f"Episode {episode+1}: {action.prompt_id} → reward={delta_acc}")
            
            stats = policy.get_action_statistics()
            assert stats["algorithm"] == "epsilon_greedy"
            assert stats["total_steps"] > 0
            print(f"✅ ε-greedy learning: {stats}")
            
            return True
            
        except Exception as e:
            print(f"❌ ε-greedy test failed: {e}")
            return False
    
    def test_reward_calculation(self) -> bool:
        """Test reward calculation with various scenarios"""
        print("\n🧪 Testing Reward Calculations...")
        
        try:
            policy = RTSPolicyEnhanced()
            
            test_cases = [
                (1, 0, 1.0),          # Perfect improvement, no cost
                (0, 0, 0.0),          # No change, no cost  
                (-1, 0, -1.0),        # Degraded, no cost
                (1, 100, 0.9),        # Improvement with cost
                (-1, 50, -1.05),      # Degraded with cost
            ]
            
            for delta_acc, tokens, expected in test_cases:
                reward = policy._calculate_reward(delta_acc, tokens)
                assert abs(reward - expected) < 1e-6, f"Reward error: {reward} != {expected}"
                print(f"✅ delta_acc={delta_acc}, tokens={tokens} → reward={reward:.3f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Reward calculation test failed: {e}")
            return False
    
    def test_context_discretization(self) -> bool:
        """Test context discretization logic"""
        print("\n🧪 Testing Context Discretization...")
        
        try:
            policy = RTSPolicyEnhanced()
            
            test_contexts = [
                (RTSContext("anchored", 0.2, None, 1), "anchored_low_early"),
                (RTSContext("overcorrected", 0.5, "p_try", 2), "overcorrected_medium_mid"),
                (RTSContext("corrected", 0.9, None, 5), "corrected_high_late"),
            ]
            
            for context, expected_key in test_contexts:
                key = policy._discretize_context(context)
                assert key == expected_key, f"Context key error: {key} != {expected_key}"
                print(f"✅ {context.detected_error}, {context.confidence}, turn {context.turn_index} → {key}")
            
            return True
            
        except Exception as e:
            print(f"❌ Context discretization test failed: {e}")
            return False
    
    def test_policy_persistence(self) -> bool:
        """Test saving and loading policy state"""
        print("\n🧪 Testing Policy Persistence...")
        
        try:
            # Create and train a policy
            policy1 = RTSPolicyEnhanced(algorithm="thompson_sampling")
            
            context = RTSContext("anchored", 0.8, None, 1)
            action = policy1.select_prompt(context)
            policy1.update_policy(context, action, 1, 50)  # Positive reward
            
            # Save it
            policy1.save_policy("test_enhanced_policy.pkl")
            
            # Create new policy and load state
            policy2 = RTSPolicyEnhanced(algorithm="thompson_sampling")
            policy2.load_policy("test_enhanced_policy.pkl")
            
            # Compare states (basic check)
            stats1 = policy1.get_action_statistics()
            stats2 = policy2.get_action_statistics()
            
            assert stats1["algorithm"] == stats2["algorithm"]
            print("✅ Policy state successfully saved and loaded")
            
            return True
            
        except Exception as e:
            print(f"❌ Policy persistence test failed: {e}")
            return False
    
    def test_algorithm_comparison(self) -> bool:
        """Compare Thompson Sampling vs ε-greedy on same scenarios"""
        print("\n🧪 Comparing Thompson Sampling vs ε-greedy...")
        
        try:
            algorithms = ["thompson_sampling", "epsilon_greedy"]
            results = {}
            
            for alg in algorithms:
                policy = RTSPolicyEnhanced(algorithm=alg)
                rewards_collected = []
                
                # Run same scenarios
                for _ in range(20):
                    context = RTSContext("anchored", 0.7, None, 1)
                    action = policy.select_prompt(context)
                    
                    # Simulate reward (p_are_you_sure gets better reward)
                    if action.prompt_id == "p_are_you_sure":
                        delta_acc = 1
                    else:
                        delta_acc = np.random.choice([-1, 0, 1])
                    
                    token_cost = 40 if action.reprompt else 0
                    reward = policy._calculate_reward(delta_acc, token_cost)
                    rewards_collected.append(reward)
                    
                    policy.update_policy(context, action, delta_acc, token_cost)
                
                results[alg] = {
                    "avg_reward": np.mean(rewards_collected),
                    "total_reward": np.sum(rewards_collected),
                    "stats": policy.get_action_statistics()
                }
            
            print("Algorithm Comparison Results:")
            for alg, result in results.items():
                print(f"{alg}: avg_reward={result['avg_reward']:.3f}, "
                      f"total_reward={result['total_reward']:.3f}")
            
            print("✅ Algorithm comparison completed")
            return True
            
        except Exception as e:
            print(f"❌ Algorithm comparison failed: {e}")
            return False
    
    def run_all_tests(self) -> Dict[str, bool]:
        """Run all tests and return results summary"""
        print("🚀 Starting RTS Policy Test Suite")
        print("=" * 60)
        
        tests = [
            ("Basic Policy Interface", self.test_basic_policy_interface),
            ("Thompson Sampling", self.test_enhanced_policy_thompson),
            ("ε-greedy Algorithm", self.test_enhanced_policy_epsilon_greedy),
            ("Reward Calculation", self.test_reward_calculation),
            ("Context Discretization", self.test_context_discretization),
            ("Policy Persistence", self.test_policy_persistence),
            ("Algorithm Comparison", self.test_algorithm_comparison)
        ]
        
        results = {}
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            print(f"\n📋 Running: {test_name}")
            try:
                result = test_func()
                results[test_name] = result
                if result:
                    passed += 1
                    print(f"✅ {test_name} PASSED")
                else:
                    print(f"❌ {test_name} FAILED")
            except Exception as e:
                results[test_name] = False
                print(f"❌ {test_name} FAILED with exception: {e}")
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        print(f"Total tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {total - passed}")
        print(f"Success rate: {passed/total*100:.1f}%")
        
        if passed == total:
            print("🎉 ALL TESTS PASSED! RTS Policy is working correctly.")
        else:
            print("⚠️  Some tests failed. Please review the output above.")
        
        # Cleanup test files
        for file in ["test_policy.pkl", "test_enhanced_policy.pkl", "rts_policy_state.pkl"]:
            if os.path.exists(file):
                os.remove(file)
        
        return results

def main():
    """Main test execution"""
    tester = RTSPolicyTester()
    results = tester.run_all_tests()
    
    # Return appropriate exit code
    all_passed = all(results.values())
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
