#!/usr/bin/env bash
#
# DIAGNOSTIC & FIX SCRIPT for LearnerBot API Integration
# - Safely diagnoses and patches the non-functional OpenAI path.
# - Adds minimal, real OpenAI support and debug logging.
# - Ensures API keys are never leaked or committed.
# - Designed to be run as a file, avoiding Warp AI agent errors.
#
set -Eeuo pipefail

# Friendly error reporting
trap 'echo "🛑 ERROR: Script failed near line $LINENO. Check the logs above for details."' ERR

# --- Main Functions ---

preflight_checks() {
    echo "=== 0) Preflight & .env Safety ==="
    git checkout pivot/teacher-learner-rts
    
    # Ensure .env is properly ignored
    touch .gitignore
    grep -qE '(^|/)\.env$' .gitignore || echo ".env" >> .gitignore
    git rm --cached .env >/dev/null 2>&1 || true
    
    # Install dependencies and create directories
    [ -f requirements.txt ] && pip install -q -r requirements.txt
    mkdir -p outputs src/agents
    echo "✅ Preflight checks passed."
}

diagnose_stub() {
    echo "=== 1) Diagnosis: Checking for stub code in LearnerBot ==="
    # Look for a non-demo 'else' returning a literal '0'
    ( git grep -nE 'class\s+LearnerBot|def\s+answer|provider.*openai|return\s+\"?0\"?' -- src || true ) | sed -n '1,200p'
}

patch_source_files() {
    echo "=== 2) Patch: Implementing real OpenAI path and logging ==="
    
    # Patch 1: Implement the OpenAI path in LearnerBot
    python - << 'PY'
from pathlib import Path
# Define the full, correct code for the LearnerBot
SKELETON = r'''
import os, re, json
from typing import Dict, Any, Tuple, List

def _first_number(s: str) -> str | None:
    """Safely extracts the first floating-point number from a string."""
    m = re.search(r'[-+]?\d+(?:\.\d+)?', s)
    return m.group(0) if m else None

class LearnerBot:
    def __init__(self, provider: str = "demo", temperature: float = 0.3, model: str | None = None):
        self.provider = provider
        self.temperature = temperature
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def answer(self, question: str, history: List[Dict[str, Any]], template: str | None = None) -> Tuple[str, float]:
        # DEMO PATH: Local arithmetic for quick tests
        if (os.getenv("DEMO_MODE", "1") == "1") or self.provider == "demo":
            expr = re.sub(r'[^0-9+\\-*/. ]', '', question)
            if re.search(r'[0-9]', expr) and re.search(r'[+\\-*/]', expr):
                try:
                    val = eval(expr)
                    out = str(int(val)) if float(val).is_integer() else f"{val:.4f}".rstrip('0').rstrip('.')
                    return out, 0.95
                except Exception:
                    pass
            return "0", 0.3

        # OPENAI PATH: Real API call for number-only replies
        if self.provider == "openai":
            try:
                from openai import OpenAI
                client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

                sys_prompt = "You are a precise calculator. Return ONLY the final numeric answer. No words, no explanations, no currency symbols. Just the number."
                user_prompt = question
                if template:
                    user_prompt = f"{user_prompt}\n\n[Instruction]: {template}"

                resp = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": sys_prompt},
                              {"role": "user", "content": user_prompt}],
                    temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.2")),
                    max_tokens=20
                )
                text = (resp.choices[0].message.content or "").strip()
                num = _first_number(text)
                ans = num if num is not None else text[:24] # Fallback to raw text if no number found
                conf = 0.85 if num is not None and num == text else 0.6 # Confidence is higher for clean numeric replies

                # Safe debug logging for the first 3 interactions
                dbg_path = Path("outputs/openai_debug.json")
                blob = json.loads(dbg_path.read_text()) if dbg_path.exists() else []
                if len(blob) < 3:
                    blob.append({"question": question, "template": template, "raw_response": text, "parsed_answer": ans})
                    dbg_path.write_text(json.dumps(blob, indent=2))
                return ans, conf
            except Exception as e:
                print(f"!! OpenAI call failed: {e}", flush=True)
                return "0", 0.1 # Return low-confidence zero on error

        # Fallback for any unknown provider
        return "0", 0.3
'''
# Ensure the file is created/replaced with the correct code
Path("src/agents").mkdir(parents=True, exist_ok=True)
Path("src/agents/learner.py").write_text(SKELETON, encoding="utf-8")
print("✅ Patched src/agents/learner.py with a full implementation.")
PY

    # Patch 2: Add a mismatch logger to the accuracy function
    python - << 'PY'
from pathlib import Path
runner_path = Path("src/loop/runner.py")
if runner_path.exists():
    content = runner_path.read_text(encoding="utf-8")
    if "def accuracy(" in content and "mismatch_log" not in content:
        # Inject a logger to see why answers are marked as incorrect
        content = content.replace(
            "def accuracy(", 
            "mismatch_log = 'outputs/mismatches.log'\n\ndef accuracy("
        )
        content = content.replace(
            "return int((answer or \"\").strip() == (reference or \"\").strip())",
            "ans = (answer or \"\").strip()\n    ref = (reference or \"\").strip()\n    "
            "ok = int(ans == ref)\n    "
            "if not ok:\n        "
            "with open(mismatch_log, 'a', encoding='utf-8') as f:\n            "
            "f.write(f'MISMATCH | Parsed Answer: \"{ans}\" | Expected Reference: \"{ref}\"\\n')\n    "
            "return ok"
        )
        runner_path.write_text(content, encoding="utf-8")
        print("✅ Added mismatch logger to src/loop/runner.py.")
PY
}

rerun_pipeline() {
    echo "=== 3) Execution: Re-running pipeline with patched code ==="
    # Run Demo Mode
    export DEMO_MODE=1
    python -m src.main run --dataset data/math20.csv --max-turns 2 --out outputs/smoke_demo.json --provider demo || true
    
    # Run OpenAI mode if key exists
    if [ -f .env ]; then set -a; . ./.env; set +a; fi
    if [ -n "${OPENAI_API_KEY-}" ]; then
        export DEMO_MODE=0
        python -m src.main run --dataset data/math20.csv --max-turns 3 --out outputs/smoke_openai.json --provider openai || true
    else
        echo "⚪ No OPENAI_API_KEY detected; skipping OpenAI run."
    fi
}

show_results() {
    echo "=== 4) Results: Comparing accuracy and showing debug logs ==="
    # At-a-glance accuracy comparison
    python - << 'PY'
import json, pathlib
def get_accuracy(path):
    try:
        obj = json.load(open(path))
        return (obj.get("summary", {}) or obj).get("final_accuracy_mean", "N/A")
    except Exception:
        return "Error"

print("\n--- Accuracy Summary ---")
for name in ["smoke_demo.json", "smoke_openai.json"]:
    p = pathlib.Path("outputs") / name
    if p.exists():
        print(f"{p.name}: final_accuracy_mean = {get_accuracy(p)}")
    else:
        print(f"{p.name}: (missing)")
PY
    # Show debug logs if they were created
    echo "\n--- OpenAI Debug Log (first 3 interactions) ---"
    [ -f outputs/openai_debug.json ] && cat outputs/openai_debug.json || echo "No openai_debug.json file found."
}

commit_changes() {
    echo "=== 5) Finalize: Committing safe changes ==="
    # Final safety check to ensure .env is not committed
    git ls-files --error-unmatch .env >/dev/null 2>&1 && git rm --cached .env || true
    
    # Add the patched files and commit
    git add src/agents/learner.py src/loop/runner.py .gitignore
    git commit -m "fix(learner): implement real OpenAI path and safe debug logs" || echo "No new changes to commit."
    git diff --stat HEAD~1 || true
}


# --- Main Execution Flow ---

main() {
    preflight_checks
    diagnose_stub
    patch_source_files
    rerun_pipeline
    show_results
    commit_changes
    echo -e "\n🎉 Done. If OpenAI accuracy is still low, check 'outputs/mismatches.log' for details."
}

# Run the main function
main
