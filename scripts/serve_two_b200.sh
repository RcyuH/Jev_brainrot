#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${1:?usage: scripts/serve_two_b200.sh /path/to/openjev}"
LOG_DIR="${LOG_DIR:-artifacts/server_logs}"
mkdir -p "$LOG_DIR"

pids=()
cleanup() {
  if ((${#pids[@]})); then
    kill "${pids[@]}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

start_backend() {
  local gpu="$1"
  local port="$2"
  CUDA_VISIBLE_DEVICES="$gpu" vllm serve "$MODEL_DIR" \
    --host 127.0.0.1 \
    --served-model-name qwen \
    --port "$port" \
    --enable-prefix-caching \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.90 \
    --limit-mm-per-prompt '{"image":1}' \
    --trust-remote-code \
    --max-num-seqs 256 \
    --max-logprobs 64 \
    --gdn-prefill-backend triton \
    --quantization fp8 \
    >"$LOG_DIR/vllm-gpu${gpu}.log" 2>&1 &
  pids+=("$!")
}

wait_backend() {
  local port="$1"
  for _ in $(seq 1 180); do
    if curl --fail --silent "http://127.0.0.1:${port}/health" >/dev/null; then
      return 0
    fi
    sleep 2
  done
  echo "vLLM on port $port did not become healthy" >&2
  return 1
}

start_helper() {
  local backend_port="$1"
  local helper_port="$2"
  VLLM="http://127.0.0.1:${backend_port}/v1" \
  TOKENIZER="$MODEL_DIR" \
  READOUT_T=0.85 \
  READOUT_NOUL_T=1.829074 \
  READOUT_NOUL_BIAS=0 \
  READOUT_TARGETED=1 \
  READOUT_INSTR_STYLE=pyrepr \
  SHIM_STAGGER=1 \
  python "$MODEL_DIR/helper/shim.py" --host 127.0.0.1 --port "$helper_port" \
    >"$LOG_DIR/helper-${helper_port}.log" 2>&1 &
  pids+=("$!")
}

start_backend 0 8000
start_backend 1 8001
wait_backend 8000
wait_backend 8001
start_helper 8000 3000
start_helper 8001 3001

echo "OpenJev helpers are starting on http://127.0.0.1:3000 and :3001"
wait

