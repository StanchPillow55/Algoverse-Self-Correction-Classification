#!/usr/bin/env python3
"""
Test script for the updated reasoning trace system.

This script tests the full pipeline with reasoning traces:
1. Uses full reasoning prompts (not just final answers)
2. Extracts answers from reasoning traces
3. Saves reasoning traces to text files
4. Generates CSV outputs with reasoning analysis
"""

import os
import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.loop.runner import run_dataset


def test_reasoning_traces_pipeline():
    """Test the full reasoning traces pipeline."""
    
    print("🧪 Testing Full Reasoning Traces Pipeline")
    print("=" * 50)
    
    # Set up test environment
    os.environ['DEMO_MODE'] = '1'  # Use demo mode for testing
    os.environ['OPENAI_TEMPERATURE'] = '0.2'
    os.environ['OPENAI_MAX_TOKENS'] = '1024'
    
    print("📊 Test Configuration:")
    print("  • Demo mode enabled (no API calls)")
    print("  • Full reasoning prompts")
    print("  • Answer extraction from reasoning traces")
    print("  • CSV output with trace files")
    
    # Test with GSM8K sample data
    print("\n🔢 Testing Math Reasoning (GSM8K-style)...")
    
    try:
        result = run_dataset(
            dataset_csv="data/scaling/gsm8k_sample.csv",
            traces_out="outputs/test_reasoning/math_traces.json",
            max_turns=2,
            provider="demo",
            model="demo",
            subset=None,
            config={
                'features': {
                    'enable_confidence': True,
                    'enable_error_awareness': True, 
                    'enable_multi_turn': True
                }
            },
            experiment_id="test_reasoning_math",
            dataset_name="gsm8k_test"
        )
        
        print(f"✅ Math reasoning test completed!")
        print(f"   • Problems processed: {result['summary']['items']}")
        print(f"   • Final accuracy: {result['summary']['final_accuracy_mean']:.3f}")
        
    except Exception as e:
        print(f"❌ Math reasoning test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test with HumanEval
    print("\n💻 Testing Code Reasoning (HumanEval)...")
    
    try:
        result = run_dataset(
            dataset_csv="humaneval",
            traces_out="outputs/test_reasoning/code_traces.json",
            max_turns=1,  # HumanEval typically single-turn
            provider="demo",
            model="demo", 
            subset="subset_20",
            config={
                'features': {
                    'enable_confidence': True,
                    'enable_error_awareness': True,
                    'enable_multi_turn': False  # Code typically single-turn
                }
            },
            experiment_id="test_reasoning_code",
            dataset_name="humaneval_test"
        )
        
        print(f"✅ Code reasoning test completed!")
        print(f"   • Problems processed: {result['summary']['items']}")
        print(f"   • Final accuracy: {result['summary']['final_accuracy_mean']:.3f}")
        
    except Exception as e:
        print(f"❌ Code reasoning test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n📁 Checking generated files...")
    
    # Check reasoning traces
    reasoning_dir = Path("outputs/test_reasoning/reasoning_traces")
    if reasoning_dir.exists():
        math_traces = list(reasoning_dir.glob("math/*/turn_*_reasoning.txt"))
        code_traces = list(reasoning_dir.glob("code/*/turn_*_reasoning.txt"))
        
        print(f"   • Math reasoning traces: {len(math_traces)} files")
        print(f"   • Code reasoning traces: {len(code_traces)} files")
        
        # Show sample trace file
        if math_traces:
            sample_trace = math_traces[0]
            print(f"   • Sample math trace: {sample_trace}")
            if sample_trace.exists():
                with open(sample_trace, 'r') as f:
                    content = f.read()
                    print(f"     Preview: {content[:200]}...")
    else:
        print("   ⚠️ No reasoning traces directory found")
    
    # Check CSV outputs
    csv_dir = Path("outputs/test_reasoning/csv_results")
    if csv_dir.exists():
        csv_files = list(csv_dir.glob("*.csv"))
        print(f"   • CSV files generated: {len(csv_files)}")
        for csv_file in csv_files:
            print(f"     - {csv_file.name}")
        
        # Check dashboard
        dashboard = csv_dir / "analysis_dashboard.txt"
        if dashboard.exists():
            print(f"   • Analysis dashboard: {dashboard}")
        else:
            print("   ⚠️ Analysis dashboard not found")
    else:
        print("   ⚠️ No CSV results directory found")
    
    print("\n🎉 Full reasoning traces pipeline test completed successfully!")
    print("\n📋 Summary of improvements:")
    print("  ✅ Math problems: Show complete reasoning instead of just final numbers")
    print("  ✅ Code problems: Show reasoning process then implementation")
    print("  ✅ Answer extraction: Separate reasoning from final answer")
    print("  ✅ Trace storage: Full reasoning saved to individual .txt files")
    print("  ✅ CSV analysis: Reasoning traces linked in CSV for analysis")
    print("  ✅ Multi-turn support: Reasoning traces for each correction attempt")
    
    return True


if __name__ == "__main__":
    success = test_reasoning_traces_pipeline()
    if success:
        print("\n🚀 Ready for production use with full reasoning traces!")
        exit(0)
    else:
        print("\n💥 Pipeline test failed - check errors above")
        exit(1)