#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${1:?usage: scripts/serve_one_b200.sh /path/to/openjev}"
GPU_ID="${GPU_ID:-0}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
HELPER_PORT="${HELPER_PORT:-3000}"
LOG_DIR="${LOG_DIR:-artifacts/server_logs}"
mkdir -p "$LOG_DIR"

pids=()
cleanup() {
  if ((${#pids[@]})); then
    kill "${pids[@]}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

CUDA_VISIBLE_DEVICES="$GPU_ID" vllm serve "$MODEL_DIR" \
  --host 127.0.0.1 \
  --served-model-name qwen \
  --port "$BACKEND_PORT" \
  --enable-prefix-caching \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.90 \
  --limit-mm-per-prompt '{"image":1}' \
  --trust-remote-code \
  --max-num-seqs 256 \
  --max-logprobs 64 \
  --gdn-prefill-backend triton \
  --quantization fp8 \
  >"$LOG_DIR/vllm-gpu${GPU_ID}.log" 2>&1 &
pids+=("$!")

for _ in $(seq 1 180); do
  if curl --fail --silent "http://127.0.0.1:${BACKEND_PORT}/health" >/dev/null; then
    break
  fi
  sleep 2
done
if ! curl --fail --silent "http://127.0.0.1:${BACKEND_PORT}/health" >/dev/null; then
  echo "vLLM on port $BACKEND_PORT did not become healthy" >&2
  exit 1
fi

VLLM="http://127.0.0.1:${BACKEND_PORT}/v1" \
TOKENIZER="$MODEL_DIR" \
READOUT_T=0.85 \
READOUT_NOUL_T=1.829074 \
READOUT_NOUL_BIAS=0 \
READOUT_TARGETED=1 \
READOUT_INSTR_STYLE=pyrepr \
SHIM_STAGGER=1 \
python "$MODEL_DIR/helper/shim.py" --host 127.0.0.1 --port "$HELPER_PORT" \
  >"$LOG_DIR/helper-${HELPER_PORT}.log" 2>&1 &
pids+=("$!")

echo "OpenJev is starting on GPU $GPU_ID at http://127.0.0.1:${HELPER_PORT}"
echo "Logs: $LOG_DIR/vllm-gpu${GPU_ID}.log and $LOG_DIR/helper-${HELPER_PORT}.log"
wait
