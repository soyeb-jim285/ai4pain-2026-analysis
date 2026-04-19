#!/usr/bin/env bash
# One-command upload of the AI4Pain 2026 train/validation data to Kaggle as a
# private dataset. Stages a clean copy in /tmp so we don't pollute the repo
# with metadata files or accidentally push extra files.
#
# Behaviour:
#   - first run  -> `kaggle datasets create`
#   - subsequent -> `kaggle datasets version`
#
# Requires: KAGGLE_API_TOKEN (env var) or ~/.kaggle/access_token.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
STAGE="/tmp/ai4pain_kaggle_stage"

if [ ! -d "$REPO_ROOT/Dataset/train" ] || [ ! -d "$REPO_ROOT/Dataset/validation" ]; then
  echo "ERROR: $REPO_ROOT/Dataset/{train,validation} not found." >&2
  exit 1
fi

if ! command -v kaggle >/dev/null 2>&1; then
  echo "ERROR: kaggle CLI not on PATH. Install: 'uv tool install kaggle'." >&2
  exit 1
fi

echo "[stage] preparing $STAGE ..."
rm -rf "$STAGE"
mkdir -p "$STAGE"
cp -r "$REPO_ROOT/Dataset/train" "$STAGE/train"
cp -r "$REPO_ROOT/Dataset/validation" "$STAGE/validation"
cp "$SCRIPT_DIR/dataset-metadata.json" "$STAGE/dataset-metadata.json"

DATASET_ID="$(python3 -c "import json,sys; print(json.load(open('$STAGE/dataset-metadata.json'))['id'])")"
SLUG="${DATASET_ID#*/}"

cd "$STAGE"
echo "[push] target dataset: $DATASET_ID"

if kaggle datasets list -m -s "$SLUG" 2>/dev/null | awk 'NR>2' | grep -q "$DATASET_ID"; then
  ts="$(date -u +%FT%TZ)"
  echo "[push] dataset exists -> pushing new version ($ts)"
  kaggle datasets version -p . -m "refresh $ts" --dir-mode skip
else
  echo "[push] dataset is new  -> creating"
  kaggle datasets create -p . --dir-mode skip
fi

echo "[clean] removing $STAGE"
rm -rf "$STAGE"
echo "[done] dataset URL: https://www.kaggle.com/datasets/$DATASET_ID"
