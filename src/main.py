import argparse, os, json
import yaml
from pathlib import Path
from dotenv import load_dotenv
from src.loop.runner import run_dataset

# Load environment variables from .env file
load_dotenv()

def load_config_defaults():
    """Load defaults from configs/run.yaml if it exists"""
    config_path = Path("configs/run.yaml")
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return {
                'dataset': config.get('paths', {}).get('dataset', 'data/math20.csv'),
                'traces_out': config.get('paths', {}).get('traces_out', 'outputs/traces.json'),
                'max_turns': config.get('runtime', {}).get('max_turns', 3),
                'provider': config.get('models', {}).get('provider', 'demo'),
                'config': config  # Pass the full config
            }
        except Exception:
            pass
    return {
        'dataset': 'data/math20.csv',
        'traces_out': 'outputs/traces.json', 
        'max_turns': 3,
        'provider': 'demo',
        'config': None
    }

def determine_dataset_subsets(dataset_name):
    """Determine appropriate subset sizes for different datasets."""
    dataset_lower = dataset_name.lower()
    
    if "gsm8k" in dataset_lower:
        return ["subset_100", "subset_500", "subset_1000"]
    elif "humaneval" in dataset_lower:
        return [None]  # Full dataset (164 samples)
    else:
        # Other datasets (SuperGLUE, ToolQA, MathBench, etc.)
        return ["subset_100", "subset_500"]

def main():
    # Load config defaults
    defaults = load_config_defaults()
    
    p = argparse.ArgumentParser(prog="teacher-learner")
    sub = p.add_subparsers(dest="cmd")

    p_info = sub.add_parser("info", help="Show pipeline info")
    p_run  = sub.add_parser("run", help="Run dataset through teacher/learner loop")
    p_run.add_argument("--dataset", default=defaults['dataset'])
    p_run.add_argument("--out", default=defaults['traces_out'])
    p_run.add_argument("--max-turns", type=int, default=defaults['max_turns'])
    p_run.add_argument("--provider", default=os.getenv("PROVIDER", defaults['provider']))
    p_run.add_argument("--model", help="Specific model to use (overrides provider default)")
    p_run.add_argument("--config", help="Path to config file")
    p_run.add_argument("--subset", help="Dataset subset (for HumanEval: subset_20, subset_100, full)")
    p_run.add_argument("--auto-subsets", action="store_true", help="Automatically run all appropriate subsets for the dataset")
    
    # Resume and checkpoint options
    p_run.add_argument("--resume", action="store_true", default=True, help="Resume from existing checkpoint (default: True)")
    p_run.add_argument("--no-resume", dest="resume", action="store_false", help="Start fresh, ignore existing checkpoints")
    p_run.add_argument("--overwrite", action="store_true", help="Force fresh start, delete existing outputs")
    p_run.add_argument("--checkpoint-every", type=int, default=10, help="Save resume state every N samples")
    p_run.add_argument("--shard", help="Process only shard i/n (e.g., '2/4' for 2nd quarter)")

    args = p.parse_args()
    if args.cmd == "info":
        print("Teacher/Learner pipeline. Demo mode:", os.getenv("DEMO_MODE","1"))
    elif args.cmd == "run":
        # Load config file if specified
        config = None
        if args.config:
            try:
                with open(args.config, 'r') as f:
                    config = yaml.safe_load(f)
            except Exception as e:
                print(f"Error loading config: {e}")
                return
        elif defaults['config']:
            config = defaults['config']
            
        # Demo mode is only enabled if explicitly set
        # Remove automatic demo mode enablement
        
        # Handle overwrite flag
        if args.overwrite:
            from pathlib import Path
            output_path = Path(args.out)
            if output_path.exists():
                output_path.unlink()
                print(f"🗑️ Removed existing output: {args.out}")
            
            # Also remove checkpoint files
            checkpoint_path = output_path.with_suffix('.jsonl')
            resume_state_path = output_path.parent / f"{output_path.stem}_resume_state.json"
            for path in [checkpoint_path, resume_state_path]:
                if path.exists():
                    path.unlink()
                    print(f"🗑️ Removed checkpoint: {path}")
        
        # Add checkpoint options to config
        checkpoint_config = {
            "resume": args.resume and not args.overwrite,
            "checkpoint_every": args.checkpoint_every,
            "shard": args.shard
        }
        
        # Merge with existing config
        if config is None:
            config = {}
        config['checkpoint'] = checkpoint_config
        
        # Handle automatic subset execution
        if args.auto_subsets:
            subsets = determine_dataset_subsets(args.dataset)
            print(f"🔄 Running {len(subsets)} subsets for {args.dataset}: {subsets}")
            
            all_results = []
            for subset in subsets:
                print(f"\n🚀 Running subset: {subset or 'full'}")
                subset_out = args.out.replace('.json', f'_{subset or "full"}.json')
                
                res = run_dataset(
                    args.dataset, subset_out, args.max_turns, 
                    provider=args.provider, model=args.model, 
                    config=config, subset=subset, 
                    experiment_id=f"main_run_{subset or 'full'}", 
                    dataset_name=os.path.basename(args.dataset)
                )
                all_results.append({"subset": subset or "full", "results": res["summary"]})
                print(f"✅ Completed subset {subset or 'full'}: {res['summary']}")
            
            print("\n📊 All Subsets Summary:")
            for result in all_results:
                print(f"  {result['subset']}: {result['results']}")
        else:
            res = run_dataset(
                args.dataset, args.out, args.max_turns, 
                provider=args.provider, model=args.model, 
                config=config, subset=args.subset, 
                experiment_id="main_run", 
                dataset_name=os.path.basename(args.dataset)
            )
            print(json.dumps(res["summary"], indent=2))
    else:
        p.print_help()

if __name__ == "__main__":
    main()