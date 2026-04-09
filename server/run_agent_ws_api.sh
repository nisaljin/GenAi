#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

if [ -f ".venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

export AGENT_WS_HOST="${AGENT_WS_HOST:-0.0.0.0}"
export AGENT_WS_PORT="${AGENT_WS_PORT:-8010}"

python server/agent_ws_api.py
