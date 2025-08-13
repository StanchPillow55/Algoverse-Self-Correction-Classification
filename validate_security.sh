#!/bin/bash
# SECURITY VALIDATION: Ensure .env stays ignored and API key never committed
# - No secrets are printed (no xtrace)
# - Searches working tree, index, local history, and remote history
# - Fails fast with instructions if anything suspicious is found

set -Eeuo pipefail

echo "=== 0) Pre-flight ==="
git checkout pivot/teacher-learner-rts

echo "=== 1) Ensure .env is gitignored and NOT tracked ==="
touch .gitignore
if ! grep -qE '(^|/)\.env$' .gitignore; then
  echo ".env" >> .gitignore
  echo "Appended '.env' to .gitignore"
fi

# Unstage if it was ever tracked
git rm --cached .env >/dev/null 2>&1 || true

# Verify .env is ignored (exit nonzero if not)
if ! git check-ignore -q .env; then
  echo "ERROR: .env is not being ignored by git. Investigate .gitignore precedence."
  exit 2
fi

# Verify .env is NOT tracked
if git ls-files --error-unmatch .env >/dev/null 2>&1; then
  echo "ERROR: .env is currently tracked! Remove from index:"
  echo "  git rm --cached .env && git commit -m 'security: stop tracking .env'"
  exit 3
fi

# Optional: ensure other common env files are ignored (no commit yet)
for f in .env.local .env.development .env.production; do
  if [ -f "$f" ] && ! git check-ignore -q "$f"; then
    echo "$f" >> .gitignore
    echo "Appended '$f' to .gitignore"
  fi
done

echo "=== 2) Refresh refs (to examine remote history safely) ==="
git fetch --all -p --tags

echo "=== 3) Define safe secret patterns (no echo of actual values) ==="
# Broad OpenAI key patterns (covers 'sk-' and 'sk-proj-' variants)
PAT_KEY_1='sk-[A-Za-z0-9]{16,}'            # general fallback
PAT_KEY_2='sk-(proj|live|test)-[A-Za-z0-9_-]{16,}'  # project/live/test styled
PAT_ENV_1='OPENAI_API_KEY[[:space:]]*='     # env var assignment

echo "=== 4) Scan WORKING TREE & INDEX (no secrets printed) ==="
# grep in working tree safely (excludes .git). If matches, only show filenames/lines, not values from history patches.
set +e
MATCH_WT=0
git grep -nEI -e "$PAT_ENV_1" -e "$PAT_KEY_1" -e "$PAT_KEY_2" -- . && MATCH_WT=1 || true
set -e
if [ "$MATCH_WT" -eq 1 ]; then
  echo "ERROR: Potential secret patterns found in working tree. Inspect and remove before any commit."
  echo "TIP: search locally with your editor; never paste secrets into the shell."
  exit 10
fi

echo "=== 5) Scan FULL HISTORY (local + remotes) WITHOUT showing patches ==="
# Use --no-patch so secrets aren't echoed; only commit metadata & filenames are shown.
# If any commit IDs are listed, STOP and rotate keys + scrub history.
set +e
MATCH_HIST=0
echo "--- Matching commits for key-like strings (local + remotes) ---"
git log --all --remotes -G "$PAT_KEY_2" --no-patch --pretty='format:%H %ad %an %s' --date=iso && MATCH_HIST=1 || true
git log --all --remotes -G "$PAT_KEY_1" --no-patch --pretty='format:%H %ad %an %s' --date=iso && MATCH_HIST=1 || true
echo "--- Matching commits for OPENAI_API_KEY assignments (local + remotes) ---"
git log --all --remotes --grep "$PAT_ENV_1" --no-patch --pretty='format:%H %ad %an %s' --date=iso && MATCH_HIST=1 || true
echo "--- Commits that included a .env file (local + remotes) ---"
git log --all --remotes --no-patch --pretty='format:%H %ad %an %s' --date=iso -- .env && MATCH_HIST=1 || true
set -e

if [ "$MATCH_HIST" -ne 0 ]; then
  echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
  echo "SECURITY ALERT: One or more commits appear to contain key-like patterns,"
  echo "OPENAI_API_KEY assignments, or a tracked .env file (details above)."
  echo
  echo "IMMEDIATE ACTIONS:"
  echo "1) ROTATE the OpenAI API key in your OpenAI dashboard."
  echo "2) SCRUB history (e.g., git filter-repo or gitleaks/trufflehog) to remove secrets."
  echo "3) FORCE-PUSH only after coordinated cleanup (and after rotating the key)."
  echo "   DO NOT push until cleanup is complete."
  echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
  exit 11
fi

echo "=== 6) Optional: commit .gitignore changes (NEVER .env) ==="
git add .gitignore
git commit -m "security: ensure .env ignored and not tracked" || true
git diff --stat HEAD~1 || true

echo "=== 7) Final verification summary ==="
echo "- .env ignored?          YES"
echo "- .env tracked?          NO"
echo "- Working tree matches?  NO suspicious patterns"
echo "- History matches?       NO suspicious commits"
echo
echo "All clear. Your API key has NOT been committed or pushed (based on current refs)."

echo "
=== Tip: After you finish ===
1) Run smoke / unit tests:  pytest -q  (or  pytest tests/smoke)
2) Commit:                  git add -A && git commit -m \"<brief, present-tense summary>\"
3) Show diff:               git diff --stat HEAD~1
"
