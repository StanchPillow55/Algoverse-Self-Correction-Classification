#!/usr/bin/env bash
# Package anonymized artifacts for submission
set -euo pipefail
OUT_DIR=${1:-artifacts_anonymized}
TMP=$(mktemp -d)
mkdir -p "$OUT_DIR" "$TMP"

# Collect selected artifacts
rsync -a --exclude='.venv' --exclude='.git' \
  paper "$TMP/" 2>/dev/null || true
rsync -a --exclude='.venv' --exclude='.git' \
  configs "$TMP/" 2>/dev/null || true
rsync -a --exclude='.venv' --exclude='.git' \
  scripts "$TMP/" 2>/dev/null || true
rsync -a --exclude='.venv' --exclude='.git' \
  src "$TMP/" 2>/dev/null || true
rsync -a results "$TMP/" 2>/dev/null || true
rsync -a runs "$TMP/" 2>/dev/null || true

TAR_PATH="$OUT_DIR/artifacts.tar.gz"
mkdir -p "$OUT_DIR"
( cd "$TMP" && tar -czf "$(pwd)/../$(basename "$TAR_PATH")" . )

SHA=$(shasum -a 256 "$TAR_PATH" | awk '{print $1}')
echo "$SHA" > "$OUT_DIR/requirements.lock.hash"

echo "Packaged: $TAR_PATH"
echo "SHA256: $SHA"

