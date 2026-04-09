#!/usr/bin/env bash
set -euo pipefail

if [[ -f ".env" ]]; then
  set -a
  source .env
  set +a
fi

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
CACHE_DIR="${HF_HOME:-.hf-cache}"

# Default to Qwen2-VL 7B for speed.
# For maximum detail, set: PERCEPTION_MODEL=Qwen/Qwen2-VL-72B-Instruct-AWQ
PERCEPTION_MODEL="${PERCEPTION_MODEL:-Qwen/Qwen2-VL-7B-Instruct}"
PLANNER_MODEL="${PLANNER_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
EXECUTION_MODEL="${EXECUTION_MODEL:-facebook/audiogen-medium}"
VERIFICATION_MODEL="${VERIFICATION_MODEL:-laion/clap-htsat-fused}"

python server/multi_model_api.py \
  --host "${HOST}" \
  --port "${PORT}" \
  --cache-dir "${CACHE_DIR}" \
  --perception-model "${PERCEPTION_MODEL}" \
  --planner-model "${PLANNER_MODEL}" \
  --execution-model "${EXECUTION_MODEL}" \
  --verification-model "${VERIFICATION_MODEL}" \
  --warmup
