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
HF_HUB_VERBOSITY="${HF_HUB_VERBOSITY:-info}"
TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-info}"
PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
INSTALL_APT_DEPS="${INSTALL_APT_DEPS:-1}"
INSTALL_PIP_DEPS="${INSTALL_PIP_DEPS:-1}"
REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-server/requirements_multi_model_api.txt}"

# Default to Qwen2-VL 7B for speed.
# For maximum detail, set: PERCEPTION_MODEL=Qwen/Qwen2-VL-72B-Instruct-AWQ
PERCEPTION_MODEL="${PERCEPTION_MODEL:-Qwen/Qwen2-VL-7B-Instruct}"
PLANNER_MODEL="${PLANNER_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
EXECUTION_MODEL="${EXECUTION_MODEL:-facebook/audiogen-medium}"
VERIFICATION_MODEL="${VERIFICATION_MODEL:-laion/clap-htsat-fused}"

mkdir -p "${CACHE_DIR}"

echo "[runner] Starting multi-model API"
echo "[runner] host=${HOST} port=${PORT}"
echo "[runner] cache_dir=${CACHE_DIR}"
echo "[runner] hf_hub_verbosity=${HF_HUB_VERBOSITY} transformers_verbosity=${TRANSFORMERS_VERBOSITY}"
echo "[runner] perception_model=${PERCEPTION_MODEL}"
echo "[runner] planner_model=${PLANNER_MODEL}"
echo "[runner] execution_model=${EXECUTION_MODEL}"
echo "[runner] verification_model=${VERIFICATION_MODEL}"
echo "[runner] cache_size_before=$(du -sh "${CACHE_DIR}" 2>/dev/null | awk '{print $1}')"

if [[ "${INSTALL_APT_DEPS}" == "1" ]]; then
  if command -v apt >/dev/null 2>&1; then
    echo "[deps][apt] installing system dependencies for PyAV/ffmpeg"
    if [[ "$(id -u)" -eq 0 ]]; then
      apt update
      apt install -y pkg-config ffmpeg \
        libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev \
        libavfilter-dev libswscale-dev libswresample-dev
    elif command -v sudo >/dev/null 2>&1; then
      sudo apt update
      sudo apt install -y pkg-config ffmpeg \
        libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev \
        libavfilter-dev libswscale-dev libswresample-dev
    else
      echo "[deps][apt] skipped: need root or sudo"
    fi
  else
    echo "[deps][apt] skipped: apt not found on this host"
  fi
fi

if [[ "${INSTALL_PIP_DEPS}" == "1" ]]; then
  echo "[deps][pip] installing Python requirements from ${REQUIREMENTS_FILE}"
  python -m pip install -r "${REQUIREMENTS_FILE}"
fi

python server/multi_model_api.py \
  --host "${HOST}" \
  --port "${PORT}" \
  --cache-dir "${CACHE_DIR}" \
  --perception-model "${PERCEPTION_MODEL}" \
  --planner-model "${PLANNER_MODEL}" \
  --execution-model "${EXECUTION_MODEL}" \
  --verification-model "${VERIFICATION_MODEL}" \
  --warmup
