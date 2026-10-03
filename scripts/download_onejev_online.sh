#!/usr/bin/env bash
set -euo pipefail

BUNDLE_DIR="${1:?usage: scripts/download_onejev_online.sh /path/to/offline_bundle MODEL_REVISION CODE_REVISION}"
MODEL_REVISION="${2:?pass a pinned OmniJev/OneJev-27B commit SHA}"
CODE_REVISION="${3:?pass a pinned OmniJev/OneJev code commit SHA}"

mkdir -p "$BUNDLE_DIR/models" "$BUNDLE_DIR/sources" "$BUNDLE_DIR/wheels"

hf download OmniJev/OneJev-27B \
  --revision "$MODEL_REVISION" \
  --local-dir "$BUNDLE_DIR/models/onejev-27b"

git clone https://github.com/OmniJev/OneJev.git "$BUNDLE_DIR/sources/onejev"
git -C "$BUNDLE_DIR/sources/onejev" checkout "$CODE_REVISION"

# Build qev and download its PyTorch-serving dependencies for the offline host.
python3.12 -m pip wheel \
  --wheel-dir "$BUNDLE_DIR/wheels" \
  "$BUNDLE_DIR/sources/onejev[torch]"

find "$BUNDLE_DIR/models/onejev-27b" "$BUNDLE_DIR/sources/onejev" -type f -print0 |
  sort -z |
  xargs -0 sha256sum > "$BUNDLE_DIR/ONEJEV_SHA256SUMS"

echo "OneJev model and pinned source saved under $BUNDLE_DIR"
