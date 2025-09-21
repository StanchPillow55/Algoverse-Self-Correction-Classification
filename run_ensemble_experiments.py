#!/usr/bin/env python3
"""
Ensemble Experiment Runner
Run self-correction experiments using ensemble voting across multiple models.
"""
import os
import json
import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.ensemble.runner import run_dataset
from src.utils.api_health_monitor import APIHealthMonitor, RecoveryRecommendationEngine
from pathlib import Path
import json

def ensure_api_keys():
    """Simple API key check"""
    missing = []
    if not os.getenv("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")
    if not os.getenv("ANTHROPIC_API_KEY"):
        missing.append("ANTHROPIC_API_KEY")
    return missing

def load_ensemble_config(config_path: str) -> dict:
    """Load ensemble configuration from JSON file with error handling defaults"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Add default error handling config if not present
    if 'error_handling' not in config:
        try:
            default_error_config_path = Path(__file__).parent / "configs" / "error_handling" / "default_error_config.json"
            if default_error_config_path.exists():
                with open(default_error_config_path, 'r') as f:
                    default_config = json.load(f)
                config.update(default_config)
                print(f"📋 Using default error handling configuration")
            else:
                print(f"⚠️ No error handling configuration found, using minimal defaults")
                config['error_handling'] = {
                    'max_api_errors_per_sample': 3,
                    'max_total_api_errors': 50,
                    'max_consecutive_failures': 5,
                    'checkpoint_on_error': True
                }
        except Exception as e:
            print(f"⚠️ Failed to load default error config: {e}")
    
    return config

def run_single_ensemble_experiment(config_path: str, dataset: str, output_dir: str, 
                                 subset: str = None, experiment_id: str = None):
    """Run a single ensemble experiment with given configuration and error handling"""
    
    # Load ensemble configuration
    config = load_ensemble_config(config_path)
    config_name = Path(config_path).stem
    
    if not experiment_id:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_id = f"ensemble_{config_name}_{timestamp}"
    
    print(f"🚀 Starting ensemble experiment: {config['name']}")
    print(f"📊 Configuration: {config_path}")
    print(f"📁 Dataset: {dataset}")
    print(f"🔧 Ensemble Size: {config['ensemble_size']}")
    print(f"🎯 Models: {config.get('ensemble_models', 'Auto-configured')}")
    print(f"💾 Experiment ID: {experiment_id}")
    
    # Set up output directory
    output_path = Path(output_dir) / experiment_id
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize API health monitoring
    health_monitor = APIHealthMonitor(
        monitoring_window_hours=config.get('health_monitoring', {}).get('monitoring_window_hours', 24)
    )
    
    # Check if we should proceed based on API health
    should_pause, pause_reason = health_monitor.should_pause_experiment()
    if should_pause:
        print(f"⚠️ Experiment paused due to API health: {pause_reason}")
        print("Please check API status and try again later.")
        return {"error": "experiment_paused", "reason": pause_reason}
    
    traces_file = output_path / "traces.json"
    
    # Run the ensemble experiment
    try:
        results = run_dataset(
            dataset_csv=dataset,
            traces_out=str(traces_file),
            max_turns=config.get('experiment_settings', {}).get('max_turns', 3),
            provider=config['provider'],
            model=config.get('ensemble_models', [None])[0] if config.get('ensemble_models') else None,
            k=1,  # For HumanEval pass@k
            subset=subset,
            config=config,
            experiment_id=experiment_id,
            dataset_name=dataset
        )
        
        # Save experiment configuration and results summary
        with open(output_path / "config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        with open(output_path / "summary.json", 'w') as f:
            json.dump({
                'experiment_id': experiment_id,
                'config_file': config_path,
                'dataset': dataset,
                'subset': subset,
                'results': results.get('summary', {}),
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        # Generate health and error reports
        try:
            health_report = health_monitor.export_health_report(output_path)
            print(f"📋 API health report saved: {health_report}")
            
            # Generate recovery recommendations if there were errors
            recovery_engine = RecoveryRecommendationEngine()
            if hasattr(results, 'api_errors') and results['api_errors']:
                recovery_plan = recovery_engine.get_recovery_plan(results['api_errors'])
                with open(output_path / "recovery_plan.json", 'w') as f:
                    json.dump(recovery_plan, f, indent=2)
                print(f"🔧 Recovery plan generated: {output_path / 'recovery_plan.json'}")
                
                # Print immediate recovery actions
                if recovery_plan.get('immediate_actions'):
                    print("\n🚨 Immediate Actions Required:")
                    for action in recovery_plan['immediate_actions']:
                        print(f"  • {action}")
                        
        except Exception as e:
            print(f"⚠️ Failed to generate health/recovery reports: {e}")
        
        print(f"✅ Ensemble experiment completed successfully!")
        print(f"📁 Results saved to: {output_path}")
        print(f"🎯 Final accuracy: {results.get('summary', {}).get('final_accuracy_mean', 'N/A'):.3f}")
        
        return results
        
    except Exception as e:
        print(f"❌ Ensemble experiment failed: {e}")
        # Save error details
        with open(output_path / "error.json", 'w') as f:
            json.dump({
                'experiment_id': experiment_id,
                'error': str(e),
                'config_file': config_path,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        raise

def run_batch_ensemble_experiments(configs_dir: str, dataset: str, output_dir: str, subset: str = None, config_override_fn=None):
    """Run multiple ensemble experiments with different configurations and error handling"""
    
    configs_path = Path(configs_dir)
    if not configs_path.exists():
        raise FileNotFoundError(f"Configs directory not found: {configs_dir}")
    
    # Find all JSON config files
    config_files = list(configs_path.glob("*.json"))
    if not config_files:
        raise ValueError(f"No configuration files found in {configs_dir}")
    
    print(f"🧪 Found {len(config_files)} ensemble configurations")
    
    batch_results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for config_file in config_files:
        config_name = config_file.stem
        experiment_id = f"batch_{timestamp}_{config_name}"
        
        print(f"\n{'='*60}")
        print(f"Running ensemble configuration: {config_name}")
        print(f"{'='*60}")
        
        try:
            # Apply config override if provided
            config_path = str(config_file)
            if config_override_fn:
                # Load config, apply override, and save temporarily
                temp_config = load_ensemble_config(config_path)
                temp_config = config_override_fn(temp_config)
                temp_config_path = output_dir / f"temp_{config_name}_config.json"
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                with open(temp_config_path, 'w') as f:
                    json.dump(temp_config, f, indent=2)
                config_path = str(temp_config_path)
            
            results = run_single_ensemble_experiment(
                config_path, dataset, output_dir, subset, experiment_id
            )
            batch_results[config_name] = {
                'status': 'success',
                'results': results.get('summary', {}),
                'experiment_id': experiment_id
            }
        except Exception as e:
            print(f"❌ Failed ensemble configuration {config_name}: {e}")
            
            # Check if this was an API-related failure
            error_type = "api_error" if any(term in str(e).lower() 
                                           for term in ['api', 'rate limit', 'timeout', 'authentication']) else "other_error"
            
            batch_results[config_name] = {
                'status': 'failed',
                'error': str(e),
                'error_type': error_type,
                'experiment_id': experiment_id
            }
    
    # Save batch summary
    batch_output = Path(output_dir) / f"batch_summary_{timestamp}.json"
    with open(batch_output, 'w') as f:
        json.dump({
            'batch_id': f"batch_{timestamp}",
            'dataset': dataset,
            'subset': subset,
            'configurations': batch_results,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    
    print(f"\n🎉 Batch ensemble experiments completed!")
    print(f"📁 Batch summary: {batch_output}")
    
    # Print results summary
    successful = sum(1 for r in batch_results.values() if r['status'] == 'success')
    print(f"✅ Successful experiments: {successful}/{len(config_files)}")
    
    return batch_results

def main():
    parser = argparse.ArgumentParser(description="Run ensemble self-correction experiments")
    parser.add_argument("--config", type=str, help="Path to ensemble configuration file")
    parser.add_argument("--configs-dir", type=str, default="configs/ensemble_experiments", 
                       help="Directory containing ensemble configuration files for batch mode")
    parser.add_argument("--dataset", type=str, default="gsm8k", 
                       help="Dataset to use (gsm8k, humaneval, or CSV path)")
    parser.add_argument("--subset", type=str, help="Dataset subset (e.g., subset_20, subset_100)")
    parser.add_argument("--output-dir", type=str, default="outputs/ensemble_experiments",
                       help="Output directory for experiment results")
    parser.add_argument("--batch", action="store_true", 
                       help="Run all configurations in configs-dir")
    parser.add_argument("--demo", action="store_true", 
                       help="Run in demo mode (no API calls)")
    parser.add_argument("--error-config", type=str, 
                       help="Path to error handling configuration file")
    parser.add_argument("--error-policy", type=str, choices=["conservative", "default", "aggressive"],
                       help="Pre-defined error handling policy")
    
    args = parser.parse_args()
    
    # Set demo mode if requested
    if args.demo:
        os.environ["DEMO_MODE"] = "1"
        print("🎭 Running in DEMO mode (no API calls)")
    
    # Load error handling configuration
    error_config_override = None
    if args.error_config:
        try:
            with open(args.error_config, 'r') as f:
                error_config_override = json.load(f)
            print(f"📋 Loaded custom error handling configuration: {args.error_config}")
        except Exception as e:
            print(f"⚠️ Failed to load error config {args.error_config}: {e}")
    
    elif args.error_policy:
        policy_file = Path(__file__).parent / "configs" / "error_handling" / f"{args.error_policy}_error_config.json"
        try:
            with open(policy_file, 'r') as f:
                error_config_override = json.load(f)
            print(f"📋 Using {args.error_policy} error handling policy")
        except Exception as e:
            print(f"⚠️ Failed to load {args.error_policy} policy: {e}")
    
    # Check API keys and setup
    if not args.demo:
        missing_keys = ensure_api_keys()
        if missing_keys:
            print("⚠️  Warning: Missing API keys:", missing_keys)
            print("Some ensemble configurations may fail without proper API keys.")
            
            # Use conservative error policy if API keys are missing
            if not error_config_override:
                print("🚫 Using conservative error policy due to missing API keys")
                conservative_file = Path(__file__).parent / "configs" / "error_handling" / "conservative_error_config.json"
                try:
                    with open(conservative_file, 'r') as f:
                        error_config_override = json.load(f)
                except Exception:
                    pass
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Helper function to apply error config override
    def apply_error_config_override(config):
        if error_config_override:
            config.update(error_config_override)
            print(f"🔧 Applied error handling configuration override")
        return config
    
    if args.batch:
        # Run batch experiments
        run_batch_ensemble_experiments(args.configs_dir, args.dataset, args.output_dir, args.subset, apply_error_config_override)
    elif args.config:
        # Run single experiment
        config_path = args.config
        if error_config_override:
            # Apply override to single experiment
            temp_config = load_ensemble_config(config_path)
            temp_config = apply_error_config_override(temp_config)
            temp_config_path = Path(args.output_dir) / "temp_single_config.json"
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
            with open(temp_config_path, 'w') as f:
                json.dump(temp_config, f, indent=2)
            config_path = str(temp_config_path)
        
        run_single_ensemble_experiment(config_path, args.dataset, args.output_dir, args.subset)
    else:
        # Default: run basic OpenAI ensemble
        default_config = "configs/ensemble_experiments/openai_basic.json"
        if not Path(default_config).exists():
            print(f"❌ Default config not found: {default_config}")
            print("Please specify --config or --batch mode")
            sys.exit(1)
        
        config_path = default_config
        if error_config_override:
            temp_config = load_ensemble_config(config_path)
            temp_config = apply_error_config_override(temp_config)
            temp_config_path = Path(args.output_dir) / "temp_default_config.json"
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
            with open(temp_config_path, 'w') as f:
                json.dump(temp_config, f, indent=2)
            config_path = str(temp_config_path)
            
        run_single_ensemble_experiment(config_path, args.dataset, args.output_dir, args.subset)

if __name__ == "__main__":
    main()