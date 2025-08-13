import argparse, os, json
from src.loop.runner import run_dataset

def main():
    p = argparse.ArgumentParser(prog="teacher-learner")
    sub = p.add_subparsers(dest="cmd")

    p_info = sub.add_parser("info", help="Show pipeline info")
    p_run  = sub.add_parser("run", help="Run dataset through teacher/learner loop")
    p_run.add_argument("--dataset", default="data/math20.csv")
    p_run.add_argument("--out", default="outputs/traces.json")
    p_run.add_argument("--max-turns", type=int, default=3)
    p_run.add_argument("--provider", default=os.getenv("PROVIDER","demo"))

    args = p.parse_args()
    if args.cmd == "info":
        print("Teacher/Learner pipeline. Demo mode:", os.getenv("DEMO_MODE","1"))
    elif args.cmd == "run":
        os.environ.setdefault("DEMO_MODE", "1")
        res = run_dataset(args.dataset, args.out, args.max_turns, provider=args.provider)
        print(json.dumps(res["summary"], indent=2))
    else:
        p.print_help()

if __name__ == "__main__":
    main()
