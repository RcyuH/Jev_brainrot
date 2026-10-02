#!/usr/bin/env bash
set -euo pipefail

BUNDLE_DIR="${1:?usage: scripts/install_offline.sh /path/to/offline_bundle}"
ENV_DIR="${2:-.venv}"

python3.12 -m venv "$ENV_DIR"
"$ENV_DIR/bin/python" -m pip install --no-index \
  --find-links "$BUNDLE_DIR/wheels" \
  vllm openai httpx transformers peft \
  hatchling numpy pandas pyarrow scipy matplotlib tqdm pyyaml pytest
"$ENV_DIR/bin/python" -m pip install --no-index \
  --find-links "$BUNDLE_DIR/wheels" \
  --no-build-isolation -e .

echo "Installed into $ENV_DIR"
