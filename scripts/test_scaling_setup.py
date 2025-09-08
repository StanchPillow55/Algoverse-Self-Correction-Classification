#!/usr/bin/env python3
"""
Test script for scaling study setup
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_imports():
    """Test that all modules can be imported."""
    try:
        from utils.scaling_model_manager import ScalingModelManager
        print("✓ ScalingModelManager imported successfully")
    except Exception as e:
        print(f"✗ ScalingModelManager import failed: {e}")
        return False
    
    try:
        from data.scaling_datasets import ScalingDatasetManager
        print("✓ ScalingDatasetManager imported successfully")
    except Exception as e:
        print(f"✗ ScalingDatasetManager import failed: {e}")
        return False
    
    try:
        from experiments.scaling_runner import ScalingExperimentRunner
        print("✓ ScalingExperimentRunner imported successfully")
    except Exception as e:
        print(f"✗ ScalingExperimentRunner import failed: {e}")
        return False
    
    return True

def test_model_manager():
    """Test model manager functionality."""
    try:
        from utils.scaling_model_manager import ScalingModelManager
        manager = ScalingModelManager()
        
        # Test model listing
        models = manager.get_available_models()
        print(f"✓ Found {len(models)} available models")
        
        # Test cost estimation
        cost = manager.estimate_experiment_cost([100, 500], ["gpt-4o-mini"])
        print(f"✓ Cost estimation works: ${cost['total']:.2f}")
        
        return True
    except Exception as e:
        print(f"✗ Model manager test failed: {e}")
        return False

def test_dataset_manager():
    """Test dataset manager functionality."""
    try:
        from data.scaling_datasets import ScalingDatasetManager
        manager = ScalingDatasetManager()
        
        # Test dataset info
        info = manager.get_dataset_info("toolqa")
        print(f"✓ Dataset info retrieved: {info['name']}")
        
        return True
    except Exception as e:
        print(f"✗ Dataset manager test failed: {e}")
        return False

def main():
    print("🧪 Testing Scaling Study Setup")
    print("=" * 30)
    
    tests = [
        ("Import Tests", test_imports),
        ("Model Manager", test_model_manager),
        ("Dataset Manager", test_dataset_manager)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        if test_func():
            passed += 1
            print(f"✓ {test_name} passed")
        else:
            print(f"✗ {test_name} failed")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Setup is ready.")
    else:
        print("⚠️  Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main()
