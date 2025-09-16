#!/usr/bin/env python3
"""
Simple script to run HumanEval experiments with the self-correction system.

Usage:
    python scripts/run_humaneval_sample.py [--demo] [--subset=20]
    
Examples:
    # Run with demo mode (no API calls)
    python scripts/run_humaneval_sample.py --demo
    
    # Run with OpenAI API on subset of 20 problems
    python scripts/run_humaneval_sample.py --subset=20
    
    # Run full HumanEval dataset
    python scripts/run_humaneval_sample.py --subset=full
"""

import os
import argparse
import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.loop.runner import run_dataset
from src.data.humaneval_loader import load_humaneval_dataset, create_demo_humaneval_data


def main():
    parser = argparse.ArgumentParser(description='Run HumanEval self-correction experiments')
    parser.add_argument('--demo', action='store_true', 
                       help='Use demo mode (no API calls, uses built-in test data)')
    parser.add_argument('--subset', default='20', 
                       help='Dataset subset: "full", "20", "100" (default: 20)')
    parser.add_argument('--provider', default='openai', 
                       help='Model provider: "openai", "anthropic", or "demo" (default: openai)')
    parser.add_argument('--model', default='gpt-4o-mini',
                       help='Model name (default: gpt-4o-mini)')
    parser.add_argument('--max-turns', type=int, default=1,
                       help='Maximum turns per problem (default: 1 for HumanEval)')
    parser.add_argument('--output', default='outputs/humaneval_sample_traces.json',
                       help='Output file for traces')
    
    args = parser.parse_args()
    
    # Set up environment variables
    if args.demo:
        os.environ['DEMO_MODE'] = '1'
        provider = 'demo'
        model = 'demo'
    else:
        provider = args.provider
        model = args.model
        
        # Check for API keys
        if provider == 'openai' and not os.getenv('OPENAI_API_KEY'):
            print("Warning: OPENAI_API_KEY not set. Set it as environment variable.")
        elif provider == 'anthropic' and not os.getenv('ANTHROPIC_API_KEY'):
            print("Warning: ANTHROPIC_API_KEY not set. Set it as environment variable.")
    
    # Convert subset argument to expected format
    if args.subset in ['20', '100']:
        subset = f'subset_{args.subset}'
    else:
        subset = args.subset
    
    # Configuration for the run
    config = {
        'features': {
            'enable_confidence': True,
            'enable_error_awareness': True, 
            'enable_multi_turn': args.max_turns > 1
        }
    }
    
    print(f"Running HumanEval experiment:")
    print(f"  Provider: {provider}")
    print(f"  Model: {model}")
    print(f"  Subset: {subset}")
    print(f"  Max turns: {args.max_turns}")
    print(f"  Demo mode: {args.demo}")
    print(f"  Output: {args.output}")
    
    try:
        # Test data loading first
        if args.demo:
            print("Loading demo HumanEval data...")
            test_data = create_demo_humaneval_data()
            if subset == 'subset_20':
                test_data = test_data[:20] if len(test_data) > 20 else test_data
            elif subset == 'subset_100':  
                test_data = test_data[:100] if len(test_data) > 100 else test_data
            print(f"Loaded {len(test_data)} demo problems")
        else:
            print(f"Loading HumanEval dataset (subset: {subset})...")
            test_data = load_humaneval_dataset(subset=subset)
            print(f"Loaded {len(test_data)} problems")
        
        # Show first problem as sample
        if test_data:
            first_problem = test_data[0]
            print(f"\nSample problem:")
            print(f"  ID: {first_problem['qid']}")
            print(f"  Function: {first_problem['entry_point']}")
            print(f"  Prompt (first 100 chars): {first_problem['question'][:100]}...")
        
        # Run the experiment
        print(f"\nStarting experiment...")
        result = run_dataset(
            dataset_csv="humaneval",
            traces_out=args.output,
            max_turns=args.max_turns,
            provider=provider,
            model=model,
            subset=subset,
            config=config,
            experiment_id=f"humaneval_sample_{subset}",
            dataset_name="humaneval"
        )
        
        print(f"\nExperiment completed!")
        print(f"Results summary:")
        print(f"  Problems processed: {result['summary']['items']}")
        print(f"  Final accuracy: {result['summary']['final_accuracy_mean']:.3f}")
        print(f"  Traces saved to: {args.output}")
        
        # Show some sample results
        if result['traces']:
            print(f"\nSample results:")
            for i, trace in enumerate(result['traces'][:3]):
                final_turn = trace['turns'][-1]
                status = "✓ PASSED" if final_turn['accuracy'] else "✗ FAILED"
                print(f"  {trace['qid']}: {status}")
                if args.demo:
                    print(f"    Answer: {final_turn['answer'][:50]}...")
        
    except Exception as e:
        print(f"Error running experiment: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())