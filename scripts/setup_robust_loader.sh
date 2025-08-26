#!/usr/bin/env bash
#
# SETUP SCRIPT for Robust CSV Loading
# - Creates a reusable, caching CSV loader for local files and GitHub URLs.
# - Refactors the teacher-learner runner and GPT evaluator to use the loader.
# - Adds smoke tests for the new data loading functionality.
# - Wires up helper scripts and configs for URL-based experiments.
#
set -Eeuo pipefail

# Friendly error reporting
trap 'echo "🛑 ERROR: Script failed near line $LINENO. Check logs above for details."' ERR

# --- Main Logic ---

echo "=== 0) Pre-flight & .env Safety ==="
git checkout pivot/teacher-learner-rts
touch .gitignore
grep -qE '(^|/)\.env$' .gitignore || echo ".env" >> .gitignore
git rm --cached .env >/dev/null 2>&1 || true
git check-ignore -q .env || { echo "ERROR: .env not ignored"; exit 2; }

mkdir -p src/utils scripts configs/experiments data/cache outputs tests/smoke
echo "✅ Pre-flight checks passed."

echo "=== 1) Create Robust CSV Loader ==="
cat > src/utils/dataset_loader.py << 'PY'
import io, re, hashlib, pathlib
from typing import Optional
import pandas as pd
try:
    import requests
except ImportError:
    requests = None

BLOB_RE = re.compile(r"^https?://github\.com/([^/]+)/([^/]+)/blob/([^/]+)/(.*)$")

def to_raw(url: str) -> str:
    """Converts a GitHub 'blob' URL to a raw content URL, preserving the branch."""
    m = BLOB_RE.match(url)
    if m:
        user, repo, branch, path = m.groups()
        return f"https://raw.githubusercontent.com/{user}/{repo}/{branch}/{path}"
    return url

def read_csv_flexible(src: str, cache_dir: Optional[str] = "data/cache", **kwargs) -> pd.DataFrame:
    """Reads a CSV from a local path or URL, with support for caching."""
    kwargs.setdefault("dtype", str)
    kwargs.setdefault("keep_default_na", False)
    
    if not src.startswith("http"):
        return pd.read_csv(src, **kwargs)

    raw_url = to_raw(src)
    cache_path = None
    if cache_dir:
        h = hashlib.sha1(raw_url.encode()).hexdigest()[:16]
        cache_path = pathlib.Path(cache_dir) / f"{h}_{pathlib.Path(raw_url).name}"
        if cache_path.exists():
            return pd.read_csv(cache_path, **kwargs)

    try:
        df = pd.read_csv(raw_url, **kwargs)
    except Exception:
        if requests is None: raise ImportError("The 'requests' library is needed to fetch URLs.")
        r = requests.get(raw_url, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text), **kwargs)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)
    return df
PY
echo "✅ Created src/utils/dataset_loader.py."

echo "=== 2) Refactor Teacher–Learner Runner ==="
python - << 'PY'
from pathlib import Path
import re
p = Path("src/loop/runner.py")
if not p.exists():
    print("⚠️ src/loop/runner.py not found, skipping runner patch.")
else:
    t = p.read_text(encoding="utf-8")

    if "from src.utils.dataset_loader import read_csv_flexible" not in t:
        t = "from src.utils.dataset_loader import read_csv_flexible\n" + t

    if "QNA_MAP =" not in t:
        t = t.replace("def run_dataset(", r'''
QNA_MAP = {"qid": ["qid","id"], "question": ["question","prompt"], "reference": ["ground_truth","answer"]}
def _auto_map_row(row: dict):
    low = {str(k).lower().strip(): v for k, v in row.items()}
    def pick(keys): return next((str(low[k.lower()]) for k in keys if k.lower() in low and str(low.get(k.lower(),"")).strip()), "")
    return pick(QNA_MAP["qid"]), pick(QNA_MAP["question"]), pick(QNA_MAP["reference"])

def run_dataset(''')
        t = re.sub(r'with open\(dataset_csv,.*?\):\s*rows = .*?csv\.DictReader\(f\)\)',
                   'df = read_csv_flexible(dataset_csv)\n    rows = df.to_dict(orient="records")', t, flags=re.S)
        t = t.replace('q = row["question"]\n        ref = str(row["reference"])', '_qid, q, ref = _auto_map_row(row)')
        t = t.replace('qid = f"q{idx+1}"', 'qid = _qid or f"q{idx+1}"')
        p.write_text(t, "utf-8")
        print("✅ Patched src/loop/runner.py.")
PY

echo "=== 3) Refactor GPT Self-Correction CLI ==="
python - << 'PY'
from pathlib import Path
import re
p = Path("scripts/gpt_self_correction_eval.py")
if p.exists():
    t = p.read_text(encoding="utf-8")
    if "from src.utils.dataset_loader import read_csv_flexible" not in t:
        t = t.replace("import pandas as pd", "import pandas as pd\nfrom src.utils.dataset_loader import read_csv_flexible")
    t = re.sub(r"def _read_csv_flexible\(.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n", "def _read_csv_flexible(path_or_url: str) -> pd.DataFrame:\n    return read_csv_flexible(path_or_url)\n\n", t, flags=re.S)
    p.write_text(t, "utf-8")
    print("✅ Refactored scripts/gpt_self_correction_eval.py.")
else:
    print("⚠️ scripts/gpt_self_correction_eval.py not found, skipping.")
PY

echo "=== 4) Create Sample Local Dataset ==="
cat > data/math20.csv << 'CSV'
qid,question,ground_truth
q1,What is 2 + 3?,5
q2,What is 10 - 4?,6
q3,What is 7 * 8?,56
CSV
echo "✅ Created sample local dataset for testing."

echo "=== 5) Add Smoke Tests for Loader ==="
cat > tests/smoke/test_dataset_loader.py << 'PY'
import pytest
from src.utils.dataset_loader import read_csv_flexible

BLOB_URL = "https://github.com/StanchPillow55/Algoverse-Self-Correction-Classification/blob/feat/initial-error-table/data/ground_truth_qna.csv"

def test_local_csv_loading():
    """Test loading a local CSV file."""
    df = read_csv_flexible("data/math20.csv")
    assert not df.empty
    assert "question" in df.columns
    assert len(df) >= 1

@pytest.mark.network
def test_url_csv_loading():
    """Test loading a CSV from a GitHub URL."""
    try:
        import requests
        requests.get("https://raw.githubusercontent.com", timeout=5)
    except Exception:
        pytest.skip("Network is unreachable.")
    
    df = read_csv_flexible(BLOB_URL, cache_dir=None) # Test without caching
    assert not df.empty
    print(f"Loaded CSV with columns: {list(df.columns)}")

def test_url_to_raw_conversion():
    """Test GitHub blob URL to raw URL conversion."""
    from src.utils.dataset_loader import to_raw
    blob_url = "https://github.com/user/repo/blob/main/data/file.csv"
    raw_url = to_raw(blob_url)
    assert raw_url == "https://raw.githubusercontent.com/user/repo/main/data/file.csv"
PY

# Create pytest configuration
if [ -f pytest.ini ]; then
    if ! grep -q "markers" pytest.ini; then printf "\nmarkers =\n    network: tests that require network access\n" >> pytest.ini; fi
else
    echo -e "[pytest]\nmarkers =\n    network: tests that require network access" > pytest.ini
fi
echo "✅ Added smoke tests for the dataset loader."

echo "=== 6) Add Helper Scripts and Configs ==="
cat > scripts/fetch_datasets.py << 'PY'
#!/usr/bin/env python3
from src.utils.dataset_loader import read_csv_flexible
URL1 = "https://github.com/StanchPillow55/Algoverse-Self-Correction-Classification/blob/feat/initial-error-table/data/error_bias_examples_v3.csv"
URL2 = "https://github.com/StanchPillow55/Algoverse-Self-Correction-Classification/blob/feat/initial-error-table/data/ground_truth_qna.csv"
if __name__ == "__main__":
    print("Caching datasets from URLs...")
    try:
        read_csv_flexible(URL1, cache_dir="data/cache")
        print("✅ Cached error bias examples")
        read_csv_flexible(URL2, cache_dir="data/cache")
        print("✅ Cached ground truth QnA")
        print("✅ All datasets cached in data/cache/")
    except Exception as e:
        print(f"⚠️ Failed to cache datasets: {e}")
        exit(1)
PY
chmod +x scripts/fetch_datasets.py

cat > configs/experiments/exp_qna_urls.yaml << 'YAML'
experiments:
  - name: qna_math_openai_via_url
    dataset: https://github.com/StanchPillow55/Algoverse-Self-Correction-Classification/blob/feat/initial-error-table/data/ground_truth_qna.csv
    max_turns: 3
    provider: openai
YAML
echo "✅ Created fetch script and experiment config."

echo "=== 7) Test Installation ==="
# Don't install yet - just check requirements
echo "⚠️ Setup complete but NOT committed yet. Testing required before commit."

