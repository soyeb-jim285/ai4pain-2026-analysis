#!/usr/bin/env bash
# Upload cache/ tensors to existing soyebjim/ai4pain-2026-data dataset.
# Needed for stage2_cnn.ipynb on Kaggle.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
STAGE="/tmp/ai4pain_cache_stage"

if [ ! -d "$REPO_ROOT/cache" ]; then
  echo "ERROR: $REPO_ROOT/cache not found." >&2
  exit 1
fi

echo "[stage] preparing $STAGE ..."
rm -rf "$STAGE"
mkdir -p "$STAGE/cache"
# Copy only the 1022 tensors + meta (what the notebook needs)
cp "$REPO_ROOT/cache/train_tensor_1022.npz" "$STAGE/cache/"
cp "$REPO_ROOT/cache/train_meta_1022.parquet" "$STAGE/cache/"
cp "$REPO_ROOT/cache/validation_tensor_1022.npz" "$STAGE/cache/"
cp "$REPO_ROOT/cache/validation_meta_1022.parquet" "$STAGE/cache/"
cp "$SCRIPT_DIR/dataset-metadata.json" "$STAGE/dataset-metadata.json"

cd "$STAGE"
ts="$(date -u +%FT%TZ)"
echo "[push] dataset version: cache added ($ts)"
kaggle datasets version -p . -m "add cache tensors $ts" --dir-mode zip
echo "[done]"
rm -rf "$STAGE"
