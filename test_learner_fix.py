#!/usr/bin/env python3
"""
Test script to verify the learner bot fix works correctly.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agents.learner import LearnerBot

def test_learner_fix():
    """Test that the fixed learner bot returns full responses."""
    
    print("🧪 Testing Learner Bot Fix")
    print("=" * 40)
    
    # Test with demo mode first
    os.environ["DEMO_MODE"] = "1"
    
    learner = LearnerBot(provider="demo")
    
    test_question = "What is 2 + 3?"
    
    print(f"Question: {test_question}")
    
    try:
        # Test the new return format
        full_response, extracted_answer, confidence = learner.answer(
            test_question, 
            hist=[], 
            template=None,
            experiment_id="test",
            dataset_name="test",
            sample_id="test",
            turn_number=0
        )
        
        print(f"✅ Success! New return format works:")
        print(f"  Full Response: {full_response}")
        print(f"  Extracted Answer: {extracted_answer}")
        print(f"  Confidence: {confidence}")
        
        # Verify we get both full response and extracted answer
        if full_response and extracted_answer:
            print("✅ Both full response and extracted answer are present")
        else:
            print("❌ Missing full response or extracted answer")
            return False
            
        # Test with a more complex question
        complex_question = "Jim has 5 apples. He buys 3 more. How many apples does he have?"
        
        print(f"\nComplex Question: {complex_question}")
        
        full_response2, extracted_answer2, confidence2 = learner.answer(
            complex_question, 
            hist=[], 
            template=None,
            experiment_id="test",
            dataset_name="test", 
            sample_id="test",
            turn_number=0
        )
        
        print(f"✅ Complex question works:")
        print(f"  Full Response: {full_response2}")
        print(f"  Extracted Answer: {extracted_answer2}")
        print(f"  Confidence: {confidence2}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing learner bot: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_runner_compatibility():
    """Test that the runner can handle the new return format."""
    
    print("\n🔧 Testing Runner Compatibility")
    print("=" * 40)
    
    try:
        # Import the runner to check for syntax errors
        import loop.runner
        
        print("✅ Runner imports successfully")
        print("✅ No syntax errors in runner code")
        
        return True
        
    except Exception as e:
        print(f"❌ Error importing runner: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    
    print("🚀 Testing Learner Bot Fix")
    print("=" * 50)
    
    # Test 1: Learner bot fix
    test1_passed = test_learner_fix()
    
    # Test 2: Runner compatibility  
    test2_passed = test_runner_compatibility()
    
    print("\n" + "=" * 50)
    print("📋 Test Results:")
    print(f"  Learner Bot Fix: {'✅ PASS' if test1_passed else '❌ FAIL'}")
    print(f"  Runner Compatibility: {'✅ PASS' if test2_passed else '❌ FAIL'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 All tests passed! The fix is ready.")
        print("\nNext steps:")
        print("1. Test with a small experiment run")
        print("2. Verify full reasoning traces are saved")
        print("3. Rerun experiments with the fixed pipeline")
    else:
        print("\n❌ Some tests failed. Please fix the issues before proceeding.")
    
    return test1_passed and test2_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
