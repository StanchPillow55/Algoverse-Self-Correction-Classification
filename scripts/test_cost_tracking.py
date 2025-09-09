#!/usr/bin/env python3
"""
Test cost tracking functionality
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_cost_tracking():
    """Test cost tracking functionality."""
    print("🧪 Testing Cost Tracking")
    print("=" * 30)
    
    try:
        from src.utils.cost_tracker import CostTracker, record_cost, get_cost_tracker
        
        # Record some test usage using the global tracker
        record_cost("gpt-4o-mini", "openai", 100, 50, "test_exp", "test_dataset", "test_sample", 0)
        record_cost("claude-haiku", "anthropic", 150, 75, "test_exp", "test_dataset", "test_sample", 1)
        
        print("✓ Cost records added")
        
        # Get the global tracker and summary
        tracker = get_cost_tracker()
        summary = tracker.get_summary()
        print(f"✓ Total cost: ${summary['total_cost']:.4f}")
        print(f"✓ Total tokens: {summary['total_tokens']}")
        
        # Print summary
        tracker.print_summary()
        
        # Save records
        filepath = tracker.save_records("test_cost_records.json")
        print(f"✓ Records saved to: {filepath}")
        
        return True
        
    except Exception as e:
        print(f"✗ Cost tracking test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_cost_tracking()
    if success:
        print("\n🎉 Cost tracking test passed!")
    else:
        print("\n❌ Cost tracking test failed!")
