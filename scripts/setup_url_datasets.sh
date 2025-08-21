#!/usr/bin/env bash
#
# SETUP SCRIPT for URL-based Datasets
# - Adds a flexible CSV loader for local paths or GitHub URLs.
# - Patches the main runner to auto-map CSV headers.
# - Adds configs and helper scripts for the new datasets.
# - Updates documentation with new usage instructions.
#
set -Eeuo pipefail

# Friendly error reporting
trap 'echo "🛑 ERROR: Script failed near line $LINENO. Check logs for details."' ERR

# --- Main Logic ---

echo "=== 0) Pre-flight & .env Hygiene ==="
git checkout pivot/teacher-learner-rts
touch .gitignore
grep -qE '(^|/)\.env$' .gitignore || echo ".env" >> .gitignore
git rm --cached .env >/dev/null 2>&1 || true
git check-ignore -q .env || { echo "ERROR: .env not ignored."; exit 2; }
mkdir -p outputs scripts configs/experiments data/cache
echo "✅ Pre-flight checks passed."

echo "=== 1) Add Flexible CSV Loader ==="
mkdir -p src/utils
cat > src/utils/dataset_loader.py << 'PY'
import io, os, re
from typing import Optional
import pandas as pd
try:
    import requests
except ImportError:
    requests = None

BLOB_RE = re.compile(r"^https?://github\.com/([^/]+)/([^/]+)/blob/([^/]+)/(.*)")

def _to_raw(url: str) -> str:
    m = BLOB_RE.match(url)
    if m:
        user, repo, branch, path = m.groups()
        return f"https://raw.githubusercontent.com/{user}/{repo}/{branch}/{path}"
    return url

def read_csv_flexible(src: str, cache_dir: Optional[str] = None) -> pd.DataFrame:
    if not src.startswith("http"):
        return pd.read_csv(src, dtype=str, keep_default_na=False)

    raw_url = _to_raw(src)
    if cache_dir:
        import hashlib, pathlib
        h = hashlib.sha1(raw_url.encode()).hexdigest()[:16]
        cache_path = pathlib.Path(cache_dir) / f"{h}_{pathlib.Path(raw_url).name}"
        if cache_path.exists():
            return pd.read_csv(cache_path, dtype=str, keep_default_na=False)

    try:
        df = pd.read_csv(raw_url, dtype=str, keep_default_na=False)
    except Exception:
        if requests is None: raise ImportError("requests library is required for fetching URLs.")
        r = requests.get(raw_url, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text), dtype=str, keep_default_na=False)

    if cache_dir:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)
    return df
PY
echo "✅ Created src/utils/dataset_loader.py."

echo "=== 2) Patch Runner for QnA CSVs ==="
python - << 'PY'
from pathlib import Path
import re
p = Path("src/loop/runner.py")
if not p.exists():
    print("⚪ runner.py not found, skipping patch.")
else:
    t = p.read_text(encoding="utf-8")
    if "read_csv_flexible" not in t:
        t = "from src.utils.dataset_loader import read_csv_flexible\n" + t
    
    if "QNA_MAP" not in t:
        inject = r'''
QNA_MAP = {"qid": ["qid","id"], "question": ["question","prompt"], "reference": ["ground_truth","answer"]}
def _auto_map_row(row: dict) -> tuple[str, str, str]:
    low = {str(k).lower().strip(): v for k, v in row.items()}
    def pick(keys): return next((str(low[k.lower()]) for k in keys if k.lower() in low and str(low.get(k.lower(),"")).strip()), "")
    return pick(QNA_MAP["qid"]), pick(QNA_MAP["question"]), pick(QNA_MAP["reference"])
'''
        t = t.replace("def run_dataset(", inject + "\n\ndef run_dataset(")
        t = re.sub(r'with open\(dataset_csv.*?as f\):\s*rows = list\(csv\.DictReader\(f\)\)',
                   'df = read_csv_flexible(dataset_csv, cache_dir="data/cache")\n    rows = df.to_dict(orient="records")', t, flags=re.S)
        t = t.replace('q = row["question"]\n        ref = str(row["reference"])', 'qid_m, q, ref = _auto_map_row(row)')
        t = t.replace('qid = f"q{idx+1}"', 'qid = qid_m or f"q{idx+1}"')
        p.write_text(t, encoding="utf-8")
        print("✅ Patched src/loop/runner.py with URL support and header mapping.")
PY

echo "=== 3) Add Dataset Configs and Fetcher ==="
cat > configs/experiments/datasets.yaml << 'YAML'
datasets:
  error_lib_url: https://github.com/StanchPillow55/Algoverse-Self-Correction-Classification/blob/feat/initial-error-table/data/error_bias_examples_v3.csv
  qna_math_url:  https://github.com/StanchPillow55/Algoverse-Self-Correction-Classification/blob/feat/initial-error-table/data/ground_truth_qna.csv
YAML
cat > scripts/fetch_datasets.py << 'PY'
#!/usr/bin/env python3
import argparse, pathlib, yaml
from src.utils.dataset_loader import read_csv_flexible
if __name__ == "__main__":
    with open("configs/experiments/datasets.yaml", "r") as f:
        cfg = yaml.safe_load(f)["datasets"]
    print("Fetching and caching datasets...")
    read_csv_flexible(cfg["error_lib_url"], cache_dir="data/cache")
    read_csv_flexible(cfg["qna_math_url"], cache_dir="data/cache")
    print("✅ Datasets cached in data/cache/.")
PY
chmod +x scripts/fetch_datasets.py
echo "✅ Created dataset configs and fetcher script."

echo "=== 4) Add Experiment Config and Runner ==="
cat > configs/experiments/exp_qna_urls.yaml << 'YAML'
experiments:
  - name: qna_math_openai_T3_k3
    dataset: https://github.com/StanchPillow55/Algoverse-Self-Correction-Classification/blob/feat/initial-error-table/data/ground_truth_qna.csv
    max_turns: 3
    provider: openai
YAML
cat > scripts/run_experiments.py << 'PY'
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
PY
chmod +x scripts/run_experiments.py
cat > scripts/run_self_correction_eval.sh << 'SH'
#!/usr/bin/env bash
set -Eeuo pipefail
mkdir -p outputs
python scripts/gpt_self_correction_eval.py \
  --error-csv "https://github.com/StanchPillow55/Algoverse-Self-Correction-Classification/blob/main/data/error_bias_examples_v3.csv" \
  --qna-csv   "https://github.com/StanchPillow55/Algoverse-Self-Correction-Classification/blob/main/data/ground_truth_qna.csv" \
  > outputs/self_correction.jsonl
echo "Wrote outputs/self_correction.jsonl"
SH
chmod +x scripts/run_self_correction_eval.sh
echo "✅ Created experiment config and runners."

echo "=== 5) Update README ==="
awk '1; END{
print "\n## Datasets via GitHub URLs\n";
print "The pipeline can now run directly from the ground-truth CSVs hosted on GitHub.\n";
print "### Run teacher–learner pipeline on the QnA URL:";
print "```bash";
print "# This requires OPENAI_API_KEY to be set in .env";
print "python scripts/run_experiments.py --config configs/experiments/exp_qna_urls.yaml";
print "```";
print "\n### Run GPT-4 self-correction evaluator:";
print "```bash";
print "scripts/run_self_correction_eval.sh";
print "```";
}' README.md > README.md.new && mv README.md.new README.md
echo "✅ Patched README.md with new instructions."

echo "=== 6) Finalizing and Committing ==="
git rm --cached .env >/dev/null 2>&1 || true # Final safety check
git add -A
git commit -m "feat: add URL-based dataset support with caching and header mapping" || echo "⚪ No new changes to commit."
git diff --stat HEAD~1 || true

echo -e "\n🎉 Setup complete. You can now run experiments using URLs."
echo "See the updated README.md for run commands."

