#!/usr/bin/env python3
"""
Run all scaling study experiments (Phase 1, 2, and 3)
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def run_phase_experiments(phase: int, output_dir: str = "outputs/scaling_experiments") -> Dict[str, Any]:
    """Run experiments for a specific phase."""
    
    print(f"\n🚀 Running Phase {phase} Experiments")
    print("=" * 40)
    
    # Phase configurations
    phase_configs = {
        1: {
            "description": "Validation phase - 2 models, 1 dataset, 100 samples",
            "models": ["gpt-4o-mini", "claude-haiku"],
            "datasets": ["toolqa"],
            "sample_size": 100,
            "max_turns": 3
        },
        2: {
            "description": "Medium scale - 4 models, 2 datasets, 500 samples",
            "models": ["gpt-4o-mini", "claude-haiku", "gpt-4o", "claude-sonnet"],
            "datasets": ["toolqa", "superglue"],
            "sample_size": 500,
            "max_turns": 3
        },
        3: {
            "description": "Full scale - 7 models, 5 datasets, 1000 samples",
            "models": ["gpt-4o-mini", "claude-haiku", "gpt-4o", "claude-sonnet", "llama-70b", "gpt-4", "claude-opus"],
            "datasets": ["toolqa", "superglue", "mathbench", "humaneval", "gsm8k"],
            "sample_size": 1000,
            "max_turns": 3
        }
    }
    
    if phase not in phase_configs:
        raise ValueError(f"Invalid phase: {phase}. Must be 1, 2, or 3.")
    
    config = phase_configs[phase]
    print(f"Phase {phase}: {config['description']}")
    print(f"Models: {', '.join(config['models'])}")
    print(f"Datasets: {', '.join(config['datasets'])}")
    print(f"Sample size: {config['sample_size']}")
    
    # Create output directory
    phase_output_dir = Path(output_dir) / f"phase{phase}"
    phase_output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "phase": phase,
        "config": config,
        "start_time": datetime.now().isoformat(),
        "experiments": [],
        "summary": {}
    }
    
    total_experiments = len(config['models']) * len(config['datasets'])
    experiment_count = 0
    
    # Run experiments
    for model in config['models']:
        for dataset in config['datasets']:
            experiment_count += 1
            print(f"\n  Experiment {experiment_count}/{total_experiments}: {model} on {dataset}")
            
            try:
                # Determine dataset file
                dataset_file = f"data/scaling/{dataset}_sample_{config['sample_size']}.csv"
                
                if not Path(dataset_file).exists():
                    print(f"    ⚠️  Dataset file not found: {dataset_file}")
                    continue
                
                # Run experiment
                experiment_result = run_single_experiment(
                    model=model,
                    dataset_file=dataset_file,
                    max_turns=config['max_turns'],
                    output_dir=str(phase_output_dir)
                )
                
                results['experiments'].append(experiment_result)
                print(f"    ✅ Completed: {experiment_result.get('status', 'unknown')}")
                
            except Exception as e:
                print(f"    ❌ Failed: {e}")
                results['experiments'].append({
                    "model": model,
                    "dataset": dataset,
                    "status": "failed",
                    "error": str(e)
                })
    
    # Calculate summary
    results['end_time'] = datetime.now().isoformat()
    results['summary'] = calculate_phase_summary(results)
    
    # Save results
    results_file = phase_output_dir / f"phase{phase}_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✅ Phase {phase} complete!")
    print(f"   Results saved to: {results_file}")
    print(f"   Successful experiments: {results['summary']['successful']}/{total_experiments}")
    
    return results

def run_single_experiment(model: str, dataset_file: str, max_turns: int, output_dir: str) -> Dict[str, Any]:
    """Run a single experiment."""
    
    # Determine provider from model name
    provider_mapping = {
        "gpt-4o-mini": "openai",
        "gpt-4o": "openai", 
        "gpt-4": "openai",
        "claude-haiku": "anthropic",
        "claude-sonnet": "anthropic",
        "claude-opus": "anthropic",
        "llama-70b": "replicate"
    }
    
    provider = provider_mapping.get(model, "openai")
    
    # Create experiment ID
    experiment_id = f"{model}_{Path(dataset_file).stem}_{int(time.time())}"
    
    # Run the experiment using the main pipeline
    try:
        from src.loop.runner import run_dataset
        
        result = run_dataset(
            dataset_csv=dataset_file,
            traces_out=f"{output_dir}/{experiment_id}_traces.json",
            max_turns=max_turns,
            provider=provider,
            model=model,
            experiment_id=experiment_id,
            dataset_name=Path(dataset_file).stem
        )
        
        return {
            "model": model,
            "provider": provider,
            "dataset_file": dataset_file,
            "experiment_id": experiment_id,
            "status": "success",
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "model": model,
            "provider": provider,
            "dataset_file": dataset_file,
            "experiment_id": experiment_id,
            "status": "failed",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

def calculate_phase_summary(results: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate summary statistics for a phase."""
    
    experiments = results.get('experiments', [])
    successful = [e for e in experiments if e.get('status') == 'success']
    failed = [e for e in experiments if e.get('status') == 'failed']
    
    summary = {
        "total_experiments": len(experiments),
        "successful": len(successful),
        "failed": len(failed),
        "success_rate": len(successful) / len(experiments) if experiments else 0,
        "models_tested": len(set(e['model'] for e in experiments)),
        "datasets_tested": len(set(e['dataset_file'] for e in experiments)),
        "duration_minutes": 0  # Will be calculated
    }
    
    # Calculate duration
    if 'start_time' in results and 'end_time' in results:
        start = datetime.fromisoformat(results['start_time'])
        end = datetime.fromisoformat(results['end_time'])
        duration = (end - start).total_seconds() / 60
        summary['duration_minutes'] = round(duration, 2)
    
    return summary

def run_all_phases(output_dir: str = "outputs/scaling_experiments") -> Dict[str, Any]:
    """Run all phases of the scaling study."""
    
    print("🔬 Scaling Study: Complete Experiment Suite")
    print("=" * 50)
    print(f"Output directory: {output_dir}")
    print(f"Start time: {datetime.now().isoformat()}")
    
    all_results = {
        "study_info": {
            "name": "Scaling Laws for Self-Correction in Large Language Models",
            "start_time": datetime.now().isoformat(),
            "total_phases": 3
        },
        "phases": {},
        "overall_summary": {}
    }
    
    # Run each phase
    for phase in [1, 2, 3]:
        try:
            phase_results = run_phase_experiments(phase, output_dir)
            all_results['phases'][f'phase{phase}'] = phase_results
            
        except Exception as e:
            print(f"❌ Phase {phase} failed: {e}")
            all_results['phases'][f'phase{phase}'] = {
                "phase": phase,
                "status": "failed",
                "error": str(e)
            }
    
    # Calculate overall summary
    all_results['overall_summary'] = calculate_overall_summary(all_results)
    all_results['study_info']['end_time'] = datetime.now().isoformat()
    
    # Save overall results
    results_file = Path(output_dir) / "complete_study_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n🎉 Complete study finished!")
    print(f"   Results saved to: {results_file}")
    print(f"   Total experiments: {all_results['overall_summary']['total_experiments']}")
    print(f"   Success rate: {all_results['overall_summary']['success_rate']:.1%}")
    
    return all_results

def calculate_overall_summary(all_results: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate overall summary for all phases."""
    
    total_experiments = 0
    total_successful = 0
    total_failed = 0
    
    for phase_key, phase_data in all_results.get('phases', {}).items():
        if isinstance(phase_data, dict) and 'summary' in phase_data:
            summary = phase_data['summary']
            total_experiments += summary.get('total_experiments', 0)
            total_successful += summary.get('successful', 0)
            total_failed += summary.get('failed', 0)
    
    return {
        "total_experiments": total_experiments,
        "total_successful": total_successful,
        "total_failed": total_failed,
        "success_rate": total_successful / total_experiments if total_experiments > 0 else 0,
        "phases_completed": len([p for p in all_results.get('phases', {}).values() if isinstance(p, dict) and p.get('status') != 'failed'])
    }

def main():
    """Main function to run all experiments."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run all scaling study experiments')
    parser.add_argument('--phases', nargs='+', type=int, choices=[1, 2, 3], default=[1, 2, 3],
                       help='Phases to run (default: all)')
    parser.add_argument('--output-dir', default='outputs/scaling_experiments',
                       help='Output directory for results')
    parser.add_argument('--phase-only', type=int, choices=[1, 2, 3],
                       help='Run only a specific phase')
    
    args = parser.parse_args()
    
    try:
        if args.phase_only:
            # Run single phase
            results = run_phase_experiments(args.phase_only, args.output_dir)
        else:
            # Run specified phases
            if set(args.phases) == {1, 2, 3}:
                # Run all phases
                results = run_all_phases(args.output_dir)
            else:
                # Run specific phases
                all_results = {
                    "study_info": {
                        "name": "Scaling Laws for Self-Correction in Large Language Models",
                        "start_time": datetime.now().isoformat(),
                        "phases_run": args.phases
                    },
                    "phases": {},
                    "overall_summary": {}
                }
                
                for phase in args.phases:
                    phase_results = run_phase_experiments(phase, args.output_dir)
                    all_results['phases'][f'phase{phase}'] = phase_results
                
                all_results['overall_summary'] = calculate_overall_summary(all_results)
                all_results['study_info']['end_time'] = datetime.now().isoformat()
                
                # Save results
                results_file = Path(args.output_dir) / "partial_study_results.json"
                with open(results_file, 'w') as f:
                    json.dump(all_results, f, indent=2, default=str)
                
                print(f"\n✅ Partial study complete!")
                print(f"   Results saved to: {results_file}")
                
                results = all_results
        
        print(f"\n📊 Final Summary")
        print(f"   Total experiments: {results.get('overall_summary', {}).get('total_experiments', 'N/A')}")
        print(f"   Success rate: {results.get('overall_summary', {}).get('success_rate', 0):.1%}")
        
    except Exception as e:
        print(f"❌ Error running experiments: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
