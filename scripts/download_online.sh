#!/usr/bin/env bash
set -euo pipefail

BUNDLE_DIR="${1:-offline_bundle}"
mkdir -p "$BUNDLE_DIR/models" "$BUNDLE_DIR/data" "$BUNDLE_DIR/wheels"

hf download openjev/openjev \
  --revision ac97900fd034fdd7e7e536f3d4c21b836cae0750 \
  --local-dir "$BUNDLE_DIR/models/openjev"

hf download google/civil_comments \
  --repo-type dataset \
  --revision f2970eb3a55777454c94069077cc8d9b5866312d \
  --include README.md 'data/test-*' \
  --local-dir "$BUNDLE_DIR/data/civil_comments"

hf download google-research-datasets/go_emotions \
  --repo-type dataset \
  --revision add492243ff905527e67aeb8b80c082af02207c3 \
  --include README.md 'simplified/test-*' \
  --local-dir "$BUNDLE_DIR/data/go_emotions"

python3.12 -m pip download \
  --dest "$BUNDLE_DIR/wheels" \
  --only-binary=:all: \
  --extra-index-url https://download.pytorch.org/whl/cu130 \
  'torch==2.13.0' \
  'vllm==0.29.0' \
  'transformers==5.17.0' \
  'peft==0.21.0' \
  'openai==3.16.2' \
  'httpx==0.28.1' \
  hatchling numpy pandas pyarrow scipy matplotlib tqdm pyyaml pytest

find "$BUNDLE_DIR" -type f ! -name SHA256SUMS -print0 |
  sort -z |
  xargs -0 sha256sum > "$BUNDLE_DIR/SHA256SUMS"
echo "Offline bundle created at $BUNDLE_DIR"
