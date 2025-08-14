#!/usr/bin/env bash
#
# This script moves legacy files, creates a legacy README, runs final checks,
# and safely commits the changes.
#
set -Eeuo pipefail

# Friendly error reporting for better debugging
trap 'echo "🛑 ERROR: Script failed near line $LINENO. Please check the logs above for details."' ERR

echo "=== 1) Create Legacy README ==="
mkdir -p legacy
cat > legacy/README_LEGACY.md << 'MD'
# Legacy Tree (Pre-Pivot)

This folder stores the pre-pivot classification/embeddings pipelines, tests, and datasets, mirrored under `legacy/` with their original structure (e.g., `legacy/src/classification`, `legacy/tests/unit`, `legacy/data`).

They are not executed by default and are excluded from pytest via `norecursedirs = legacy`.

If you need to run any of the legacy components, do so in an isolated virtual environment and do not mix `.env` files with the main pipeline.
MD
echo "✅ Created legacy/README_LEGACY.md"

echo "=== 2) Sanity Checks (Smoke Tests Only) ==="
pytest tests/smoke -q

echo "=== 3) Final .env Guard Before Commit ==="
# Ensure .env is not tracked or staged. This is a critical safety check.
git rm --cached .env >/dev/null 2>&1 || true

if ! git check-ignore -q .env; then
  echo "❌ ERROR: .env is not being ignored by git. Investigate .gitignore precedence."
  exit 4
fi

if git ls-files --error-unmatch .env >/dev/null 2>&1; then
  echo "❌ ERROR: .env is currently tracked by git. It must be removed from the index."
  exit 5
fi
echo "✅ .env is safely ignored and not tracked."

echo "=== 4) Commit and Show Diff ==="
git add -A
git commit -m "cleanup: migrate legacy files and update README" || echo "⚪ No new changes to commit."
git diff --stat HEAD~1 || true

echo "=== 5) Show Legacy Directory Tree ==="
find legacy -maxdepth 3 -type d | sed 's|^|  |' | sed -n '1,200p'

echo ""
echo "🎉 Cleanup script finished successfully."
echo ""
echo "---"
echo "Tip: After you finish"
echo "- Run tests:  pytest -q"
echo "- Commit:     git add -A && git commit -m \"<summary>\""
echo "- Show diff:  git diff --stat HEAD~1"
echo "---"

