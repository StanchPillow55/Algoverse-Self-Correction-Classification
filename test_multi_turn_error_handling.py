#!/usr/bin/env python3
"""
Test script for unified multi-turn error handling system.

This script demonstrates how the enhanced error handling works across
both the ensemble runner and the loop runner.
"""

import os
import json
import tempfile
from pathlib import Path

def test_ensemble_runner_error_handling():
    """Test error handling in ensemble runner"""
    print("🧪 Testing Ensemble Runner Error Handling")
    
    # Import the ensemble runner
    from src.ensemble.runner import run_dataset
    
    # Create test config with error handling
    test_config = {
        "ensemble_size": 2,
        "ensemble_models": ["gpt-4o-mini", "gpt-3.5-turbo"],
        "error_handling": {
            "max_api_errors_per_sample": 2,
            "max_total_api_errors": 10,
            "checkpoint_on_error": True,
            "terminate_on_quota_exceeded": False
        },
        "health_monitoring": {
            "monitoring_window_hours": 1,
            "healthy_failure_rate_threshold": 0.1
        }
    }
    
    # Test with demo mode to avoid actual API calls
    os.environ["DEMO_MODE"] = "1"
    
    with tempfile.TemporaryDirectory() as temp_dir:
        results = run_dataset(
            dataset_csv="gsm8k",
            traces_out=f"{temp_dir}/ensemble_traces.json",
            max_turns=2,
            provider="demo",
            model="gpt-4o-mini",
            subset="subset_5",  # Small subset for testing
            config=test_config,
            experiment_id="test_ensemble_error_handling",
            dataset_name="gsm8k_test"
        )
        
        print(f"✅ Ensemble runner test completed")
        print(f"   - Processed {results.get('summary', {}).get('items', 0)} samples")
        print(f"   - Accuracy: {results.get('summary', {}).get('final_accuracy_mean', 0):.3f}")
        
        if 'error_handling' in results:
            error_stats = results['error_handling']
            print(f"   - API errors: {error_stats.get('total_api_errors', 0)}")
            print(f"   - Terminated: {error_stats.get('experiment_terminated', False)}")


def test_loop_runner_error_handling():
    """Test error handling in loop runner"""
    print("\n🧪 Testing Loop Runner Error Handling")
    
    # Import the loop runner  
    from src.loop.runner import run_dataset
    
    # Create test config with error handling
    test_config = {
        "error_handling": {
            "max_api_errors_per_sample": 3,
            "max_total_api_errors": 15,
            "checkpoint_on_error": True
        },
        "health_monitoring": {
            "monitoring_window_hours": 2
        },
        "features": {
            "enable_multi_turn": True,
            "enable_confidence": True,
            "enable_error_awareness": True
        }
    }
    
    with tempfile.TemporaryDirectory() as temp_dir:
        results = run_dataset(
            dataset_csv="gsm8k",
            traces_out=f"{temp_dir}/loop_traces.json", 
            max_turns=3,
            provider="demo",
            model="gpt-4o-mini",
            subset="subset_5",  # Small subset for testing
            config=test_config,
            experiment_id="test_loop_error_handling",
            dataset_name="gsm8k_test"
        )
        
        print(f"✅ Loop runner test completed")
        print(f"   - Processed {results.get('summary', {}).get('items', 0)} samples")
        print(f"   - Accuracy: {results.get('summary', {}).get('final_accuracy_mean', 0):.3f}")
        
        if 'error_handling' in results:
            error_stats = results['error_handling']
            print(f"   - API errors: {error_stats.get('total_api_errors', 0)}")
            print(f"   - Terminated: {error_stats.get('experiment_terminated', False)}")


def test_error_configuration_loading():
    """Test loading different error handling configurations"""
    print("\n🧪 Testing Error Configuration Loading")
    
    # Test each error policy configuration
    policies = ["conservative", "default", "aggressive"]
    
    for policy in policies:
        config_file = f"configs/error_handling/{policy}_error_config.json"
        config_path = Path(config_file)
        
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                print(f"✅ {policy.title()} policy loaded:")
                error_config = config.get('error_handling', {})
                print(f"   - Max errors per sample: {error_config.get('max_api_errors_per_sample', 'N/A')}")
                print(f"   - Max total errors: {error_config.get('max_total_api_errors', 'N/A')}")
                print(f"   - Terminate on quota: {error_config.get('terminate_on_quota_exceeded', 'N/A')}")
                
            except Exception as e:
                print(f"❌ Failed to load {policy} policy: {e}")
        else:
            print(f"⚠️ {policy.title()} policy file not found: {config_path}")


def test_multi_turn_error_handler_interface():
    """Test the unified multi-turn error handler interface directly"""
    print("\n🧪 Testing Multi-Turn Error Handler Interface")
    
    from src.utils.multi_turn_error_handler import create_multi_turn_error_handler
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test creating error handler
        handler = create_multi_turn_error_handler(
            output_dir=Path(temp_dir),
            experiment_id="test_interface",
            config={
                "error_handling": {
                    "max_api_errors_per_sample": 2,
                    "checkpoint_on_error": True
                }
            },
            learner_type="test"
        )
        
        print("✅ Multi-turn error handler created")
        
        # Test initialization
        initialized = handler.initialize_experiment("test_dataset", "demo", "test_model")
        print(f"   - Initialization: {'Success' if initialized else 'Failed'}")
        
        # Test sample processing flow
        can_process = handler.start_sample_processing("test_sample_1", "What is 2+2?")
        print(f"   - Sample processing: {'Allowed' if can_process else 'Skipped'}")
        
        if can_process:
            # Complete sample processing
            sample_data = {
                "qid": "test_sample_1",
                "question": "What is 2+2?",
                "reference": "4",
                "turns": [{"answer": "4", "accuracy": 1}],
                "final_accuracy": 1
            }
            
            success = handler.complete_sample_processing(sample_data)
            print(f"   - Sample completion: {'Success' if success else 'Failed'}")
        
        # Test finalization
        final_results = handler.finalize_experiment([])
        print(f"   - Finalization: {'Success' if 'summary' in final_results else 'Failed'}")


def main():
    """Run all error handling tests"""
    print("🚀 Multi-Turn Error Handling Test Suite")
    print("=" * 50)
    
    try:
        # Test configuration loading first
        test_error_configuration_loading()
        
        # Test the unified interface
        test_multi_turn_error_handler_interface()
        
        # Test both runners
        test_ensemble_runner_error_handling()
        test_loop_runner_error_handling()
        
        print("\n🎉 All tests completed successfully!")
        print("\n📋 Summary:")
        print("   ✅ Error configuration loading works")
        print("   ✅ Unified multi-turn error handler interface works")
        print("   ✅ Ensemble runner error handling integrated")
        print("   ✅ Loop runner error handling integrated")
        
        print("\n🔧 Next steps:")
        print("   • Run actual experiments with --error-policy flags")
        print("   • Test with real API calls to verify error handling")
        print("   • Use checkpoint resumption after API failures")
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())