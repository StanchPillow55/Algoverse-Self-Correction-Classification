#!/usr/bin/env python3
import argparse, subprocess, sys, yaml, os, pathlib
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); args = ap.parse_args()
    with open(args.config, "r") as f: cfg = yaml.safe_load(f)
    pathlib.Path("outputs/exp").mkdir(parents=True, exist_ok=True)
    for exp in cfg.get("experiments", []):
        name, ds, mt, p = exp["name"], exp["dataset"], str(exp["max_turns"]), exp["provider"]
        out = f"outputs/exp/{name}.json"; env = os.environ.copy(); env["DEMO_MODE"] = "0"
        cmd = [sys.executable, "-m", "src.main", "run", "--dataset", ds, "--max-turns", mt, "--out", out, "--provider", p]
        print(f"=== Running experiment: {name} ==="); rc = subprocess.call(cmd, env=env)
        if rc != 0: print(f"Experiment {name} failed with rc={rc}", file=sys.stderr)
if __name__ == "__main__": main()
