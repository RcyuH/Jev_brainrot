#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${1:?usage: scripts/serve_onejev_b200.sh /path/to/onejev-27b}"
GPU_ID="${GPU_ID:-0}"

echo "Starting OneJev on GPU $GPU_ID. The qev default endpoint is http://127.0.0.1:8000"
CUDA_VISIBLE_DEVICES="$GPU_ID" qev serve --model "$MODEL_DIR" --port 8000
