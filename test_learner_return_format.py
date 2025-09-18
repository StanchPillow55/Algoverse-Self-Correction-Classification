#!/usr/bin/env python3
"""Test what format the learner bot actually returns."""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agents.learner import LearnerBot

def test_learner_return_format():
    """Test the actual return format of the learner bot."""
    
    print("🧪 Testing Learner Bot Return Format")
    print("=" * 40)
    
    # Force demo mode
    os.environ["DEMO_MODE"] = "1"
    
    learner = LearnerBot(provider="demo")
    
    test_question = "What is 2 + 3?"
    
    print(f"Question: {test_question}")
    
    try:
        result = learner.answer(test_question, hist=[], template=None)
        
        print(f"Result: {result}")
        print(f"Type: {type(result)}")
        print(f"Length: {len(result) if hasattr(result, '__len__') else 'N/A'}")
        
        if isinstance(result, tuple):
            print(f"Tuple elements:")
            for i, item in enumerate(result):
                print(f"  [{i}]: {item} (type: {type(item)})")
        
        return result
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = test_learner_return_format()
    
    if result and isinstance(result, tuple) and len(result) == 3:
        print("\n✅ SUCCESS: Learner bot returns 3-tuple (full_response, answer, confidence)")
    elif result and isinstance(result, tuple) and len(result) == 2:
        print("\n❌ PROBLEM: Learner bot returns 2-tuple (answer, confidence) - old format!")
    else:
        print(f"\n❌ UNEXPECTED: Learner bot returns {type(result)} with {len(result) if hasattr(result, '__len__') else 'unknown'} elements")
