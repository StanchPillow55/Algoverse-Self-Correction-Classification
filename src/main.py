import argparse, os, json
import yaml
from pathlib import Path
from src.loop.runner import run_dataset

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
                'provider': config.get('models', {}).get('provider', 'demo')
            }
        except Exception:
            # If config reading fails, use hardcoded defaults
            pass
    return {
        'dataset': 'data/math20.csv',
        'traces_out': 'outputs/traces.json', 
        'max_turns': 3,
        'provider': 'demo'
    }

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

    args = p.parse_args()
    if args.cmd == "info":
        print("Teacher/Learner pipeline. Demo mode:", os.getenv("DEMO_MODE","1"))
    elif args.cmd == "run":
        # Only set demo mode if no explicit setting and provider is not openai
        if "DEMO_MODE" not in os.environ and args.provider != "openai":
            os.environ.setdefault("DEMO_MODE", "1")
        res = run_dataset(args.dataset, args.out, args.max_turns, provider=args.provider)
        print(json.dumps(res["summary"], indent=2))
    else:
        p.print_help()

if __name__ == "__main__":
    main()
