#!/usr/bin/env bash
#
# REFACTOR SCRIPT: Move all non-pipeline files to a `legacy/` directory.
# - Uses an allowlist to identify and preserve the active learner-teacher pipeline.
# - Preserves file history and directory structure using `git mv`.
# - Ensures .env is and remains ignored.
#
set -Eeuo pipefail

# Friendly error reporting
trap 'echo "🛑 ERROR: Script failed near line $LINENO. Please check logs for details."' ERR

# --- Main Logic ---

echo "=== 0) Pre-flight & Safety Checks ==="
git checkout pivot/teacher-learner-rts

# Critical safety checks for .env
touch .gitignore
grep -qE '(^|/)\.env$' .gitignore || echo ".env" >> .gitignore
git rm --cached .env >/dev/null 2>&1 || true # Unstage if it was ever added
if ! git check-ignore -q .env; then
  echo "❌ ERROR: .env is not properly ignored. Please fix your .gitignore file first."
  exit 2
fi
if git ls-files --error-unmatch .env >/dev/null 2>&1; then
  echo "❌ ERROR: .env is currently tracked by Git. Please run 'git rm --cached .env' and commit."
  exit 3
fi
echo "✅ .env is safely ignored and not tracked."


echo "=== 1) Defining Pipeline Allowlist ==="
# An array of regex patterns for files/directories to KEEP. Everything else will be moved.
KEEP=(
  # Core source code for the current pipeline
  '^src/agents($|/)'
  '^src/loop($|/)'
  '^src/rts($|/)'
  '^src/main\.py$'
  # Configs, templates, and documentation
  '^configs($|/)'
  '^rts_templates\.json$'
  '^docs($|/)'
  # Active datasets
  '^data/math20\.csv$'
  '^data/facts20\.csv$'
  # Active tests
  '^tests/smoke($|/)'
  # Utility scripts and project files
  '^scripts($|/)'
  '^run_pipeline\.sh$'
  '^README\.md$'
  # Never move the destination folder or Git's own directory
  '^legacy($|/)'
  '^\.git($|/)'
  # This script file itself should not be moved during execution
  '^organize_legacy_files\.sh$'
)


echo "=== 2) Identifying Legacy Files ==="
# Create a single, combined regex from the KEEP array
KEEP_REGEX="$(IFS='|'; echo "${KEEP[*]}")"

# Get a list of all files tracked by Git
ALL_FILES=$(git ls-files)

LEGACY_LIST=()
while IFS= read -r f; do
  # Skip empty lines
  [[ -z "$f" ]] && continue
  # If a file matches the keep-list regex, skip it
  if echo "$f" | grep -Eq "$KEEP_REGEX"; then
    continue
  fi
  # Otherwise, add it to the list of files to be moved
  LEGACY_LIST+=("$f")
done <<< "$ALL_FILES"

if [ ${#LEGACY_LIST[@]} -eq 0 ]; then
  echo "✅ No legacy files to move. Project is already clean."
  exit 0
fi
echo "Found ${#LEGACY_LIST[@]} legacy file(s) to move."


echo "=== 3) Moving Legacy Files to legacy/ ==="
mkdir -p legacy
for path in "${LEGACY_LIST[@]}"; do
  # Double-check that the file exists before moving
  if [ -f "$path" ]; then
    # Ensure the destination directory exists inside legacy/
    mkdir -p "legacy/$(dirname "$path")"
    # Use git mv to preserve file history
    git mv "$path" "legacy/$path"
    echo "Moved: $path -> legacy/$path"
  fi
done


echo "=== 4) Updating pytest Configuration ==="
# Ensure pytest ignores the new legacy directory during test discovery
if [ -f pytest.ini ]; then
  if ! grep -q "^norecursedirs *= .*legacy" pytest.ini; then
    printf "\nnorecursedirs = legacy\n" >> pytest.ini
    echo "Updated pytest.ini to ignore legacy/."
  fi
else
  cat > pytest.ini << 'INI'
[pytest]
addopts = -q
testpaths = tests
filterwarnings = ignore
norecursedirs = legacy
INI
  echo "Created pytest.ini and configured it to ignore legacy/."
fi


echo "=== 5) Running Sanity Checks ==="
pytest tests/smoke -q || { echo "❌ Smoke tests failed after refactor. Please investigate before proceeding."; exit 10; }
echo "✅ Smoke tests passed."


echo "=== 6) Committing Changes ==="
git add -A
git commit -m "refactor: move legacy files into legacy/ directory" || echo "⚪ No new changes to commit."
git diff --stat HEAD~1 || true


echo -e "\n🎉 Refactoring complete. Legacy files are now in the legacy/ directory."

