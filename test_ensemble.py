#!/usr/bin/env python3
"""
Simple test script for ensemble functionality
"""
import os
import sys
import tempfile
import json
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

def test_ensemble_demo():
    """Test ensemble system in demo mode"""
    
    # Set demo mode
    os.environ["DEMO_MODE"] = "1"
    
    print("🧪 Testing ensemble system in demo mode...")
    
    try:
        from src.ensemble.learner import EnsembleLearnerBot
        from src.ensemble.metrics import EnsembleMetrics
        
        # Test ensemble learner creation
        print("✓ Creating ensemble learner...")
        learner = EnsembleLearnerBot(
            provider="demo", 
            ensemble_size=3, 
            ensemble_models=["demo-1", "demo-2", "demo-3"]
        )
        
        # Test simple math question
        print("✓ Testing ensemble answer generation...")
        question = "What is 15 + 27?"
        answer, confidence, response = learner.answer(question, [])
        
        print(f"  Question: {question}")
        print(f"  Answer: {answer}")
        print(f"  Confidence: {confidence}")
        print(f"  Response preview: {response[:100]}...")
        
        # Test voting algorithms
        print("✓ Testing voting algorithms...")
        
        # Test majority voting
        responses = ["42", "42", "43"]  # Two agree, one disagrees
        confidences = [0.8, 0.9, 0.7]
        raw_texts = ["Response 1", "Response 2", "Response 3"]
        
        final_answer, final_conf, info = learner._aggregate_ensemble_responses(
            responses, confidences, raw_texts
        )
        
        print(f"  Majority vote result: {final_answer} (confidence: {final_conf:.3f})")
        print(f"  Voting info: {info['voting_method']}")
        
        # Test weighted confidence voting
        final_answer_wc, final_conf_wc, info_wc = learner._weighted_confidence_voting(
            responses, confidences
        )
        print(f"  Weighted confidence result: {final_answer_wc} (confidence: {final_conf_wc:.3f})")
        
        # Test adaptive voting
        final_answer_ad, final_conf_ad, info_ad = learner._adaptive_voting(
            responses, confidences, raw_texts, is_code_task=False
        )
        print(f"  Adaptive voting result: {final_answer_ad} (confidence: {final_conf_ad:.3f})")
        print(f"  Chosen strategy: {info_ad.get('adaptive_strategy', 'unknown')}")
        
        # Test metrics (with dummy data)
        print("✓ Testing ensemble metrics...")
        
        # Create dummy traces data for testing
        dummy_traces = {
            "summary": {"items": 2, "final_accuracy_mean": 0.5},
            "traces": [
                {
                    "qid": "test_1",
                    "question": "Test question 1",
                    "reference": "42",
                    "turns": [
                        {
                            "answer": "42",
                            "response_text": "=== ENSEMBLE RESPONSE ===\nFinal Answer: 42\nEnsemble Confidence: 0.850\nVoting Method: majority_with_confidence\nConsensus: 3/3 models agreed\n\n=== VOTING SUMMARY ===\nResponse Distribution: {'42': 3}\nConsensus Ratio: 1.0\nAverage Individual Confidence: 0.800",
                            "accuracy": 1,
                            "combined_confidence": 0.85
                        }
                    ],
                    "final_accuracy": 1
                },
                {
                    "qid": "test_2", 
                    "question": "Test question 2",
                    "reference": "100",
                    "turns": [
                        {
                            "answer": "99",
                            "response_text": "=== ENSEMBLE RESPONSE ===\nFinal Answer: 99\nEnsemble Confidence: 0.600\nVoting Method: majority_with_confidence_tiebreak\nConsensus: 2/3 models agreed",
                            "accuracy": 0,
                            "combined_confidence": 0.60
                        }
                    ],
                    "final_accuracy": 0
                }
            ]
        }
        
        # Save dummy traces to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(dummy_traces, f, indent=2)
            temp_traces_file = f.name
        
        try:
            # Test metrics analysis
            analyzer = EnsembleMetrics()
            metrics = analyzer.analyze_ensemble_experiment(temp_traces_file)
            
            print(f"  Analyzed {metrics.get('total_questions', 0)} questions")
            basic = metrics.get('basic_metrics', {})
            print(f"  Ensemble accuracy: {basic.get('ensemble_accuracy', 0):.3f}")
            
            voting = metrics.get('voting_analysis', {})
            print(f"  Average consensus: {voting.get('average_consensus_ratio', 0):.3f}")
            
        finally:
            # Clean up temp file
            os.unlink(temp_traces_file)
        
        print("✅ All ensemble tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Ensemble test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ensemble_configs():
    """Test that ensemble configuration files are valid"""
    
    print("🧪 Testing ensemble configurations...")
    
    config_dir = Path("configs/ensemble_experiments")
    if not config_dir.exists():
        print(f"❌ Config directory not found: {config_dir}")
        return False
    
    config_files = list(config_dir.glob("*.json"))
    if not config_files:
        print("❌ No configuration files found")
        return False
    
    for config_file in config_files:
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            # Check required fields
            required_fields = ["name", "provider", "ensemble_size"]
            for field in required_fields:
                if field not in config:
                    raise ValueError(f"Missing required field: {field}")
            
            print(f"  ✓ {config_file.name}: {config['name']}")
            
        except Exception as e:
            print(f"  ❌ {config_file.name}: {e}")
            return False
    
    print(f"✅ All {len(config_files)} configuration files are valid!")
    return True

def main():
    """Run all ensemble tests"""
    
    print("=" * 60)
    print("ENSEMBLE SYSTEM TEST SUITE")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 2
    
    # Test 1: Basic functionality
    if test_ensemble_demo():
        tests_passed += 1
    
    print()
    
    # Test 2: Configuration files  
    if test_ensemble_configs():
        tests_passed += 1
    
    print()
    print("=" * 60)
    print(f"TEST RESULTS: {tests_passed}/{total_tests} tests passed")
    print("=" * 60)
    
    if tests_passed == total_tests:
        print("🎉 All ensemble tests passed! System is ready to use.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())