# GenAI Music In Video (Agentic Foley)

Agentic pipeline that generates Foley audio from video (or prompt-only), scores quality with CLAP, and iteratively retries/refines until acceptance criteria are met.

Main entrypoint:
- `main.py`: root-level orchestrator for the agentic loop.

## Prerequisites

- Python `3.10+` (tested with Python `3.12`)
- Node.js `18+` and npm
- `ffmpeg` available on `PATH` (required for robust video/audio muxing)

## Repo Structure

- `main.py`: root orchestrator and agentic loop.
- `server/`: Python backend services, helper APIs, and launch scripts.
- `web/`: Next.js frontend for upload + live event streaming.
- `notebooks/`: teammate-facing deep-dive + stage test notebook.
- `scripts/`: stage tests and helper scripts.

## Backend Entry Points

The `server/` folder contains multiple Python services:
- `agent_ws_api.py`: local WebSocket bridge and upload/artifact server.
- `multi_model_api.py`: model-serving API for perception, planning, execution, and verification.
- `server.py`: standalone audio generation/evaluation API.
- `inference_service.py`: additional inference utilities used by experiments and tests.

## Runtime Roles

In the default setup you described, these are the active pieces:
- `main.py` contains the `FoleyOrchestrator` pipeline implementation.
- `server/agent_ws_api.py` is the backend you run on the client machine; it imports `FoleyOrchestrator` from `main.py` and exposes `/upload-video`, `/artifacts/...`, and `/ws/foley`.
- `server/multi_model_api.py` runs on the server and provides the model endpoints used by the orchestrator.

The other Python files in `server/` are support or alternate entrypoints, not part of the default client/server run path:
- `server.py`: a separate FastAPI service for audio generation and CLAP evaluation.
- `inference_service.py`: helper inference utilities for experiments, tests, or alternate flows.

## Environment Setup

1. Create and activate a Python virtual environment at repo root:
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

2. Install local backend requirements (WS bridge + orchestrator) only:
```bash
python -m pip install -r server/requirements_agent_ws_api.txt
```

3. Install frontend dependencies:
```bash
cd web
npm install
cd ..
```

4. Create your runtime env file:
```bash
cp .env.example .env
```
Then edit `.env` with your own keys and endpoint values.

## Default Run Mode (Local WS/UI + Remote GPU Model API)

This is the recommended setup for your laptop:
- `server/agent_ws_api.py` runs locally.
- the heavy model API (`multi_model_api.py`) runs on a separate GPU server.
- local `.env` points `VLM_API_URL` and `AUDIO_API_URL` to that remote GPU endpoint.
- `main.py` is the root orchestrator that the WS bridge invokes when a run starts.
- `main.py` uses Groq for the planner/controller path with `llama-3.3-70b-versatile`.
- `server/multi_model_api.py` has its own planner default (`Qwen/Qwen2.5-7B-Instruct`) for the model-serving API when you run that service directly.

Run with 2 terminals:

Terminal 1 (local WS bridge):
```bash
source .venv/bin/activate
./server/run_agent_ws_api.sh
```

Terminal 2 (frontend):
```bash
cd web
npm run dev
```

Open:
`http://localhost:3000`

## Optional Run Mode (Start Multi-Model API on a GPU Host)

Only use this on the machine that should host the heavy models.

Install model-API requirements on that GPU host:
```bash
python -m pip install -r server/requirements_multi_model_api.txt
```

Then run:
```bash
./server/run_multi_model_api.sh
```

## Script Notes

- `./server/run_multi_model_api.sh`:
  - loads `.env` automatically if present,
  - installs apt deps on Linux if `apt` exists (`INSTALL_APT_DEPS=1` by default),
  - installs pip deps from `server/requirements_multi_model_api.txt` by default (`INSTALL_PIP_DEPS=1`).
- If you already installed dependencies manually, run:
```bash
INSTALL_APT_DEPS=0 INSTALL_PIP_DEPS=0 ./server/run_multi_model_api.sh
```
- `./server/run_agent_ws_api.sh` automatically activates `.venv` if available.

## Key Endpoints

- Model API health: `http://localhost:8000/health`
- WS bridge health: `http://localhost:8010/health`
- WebSocket stream: `ws://localhost:8010/ws/foley`

## Core Flow

1. Upload video (`PUT /upload-video`) or run prompt-only.
2. Start run over WebSocket (`/ws/foley`).
3. Orchestrator emits live events:
   - perception/VLM
   - planning
   - attempt loop (generate -> score -> decide)
4. Final artifact served from `/artifacts/{audio|video}/...`.
5. Frontend renders inline `<audio>`/`<video>`.

## Team Docs

- Architecture deep dive notebook:
  - `notebooks/agentic_foley_system_deep_dive.ipynb`
- Notebook index:
  - `notebooks/README.md`
- Backend details:
  - `server/README.md`
- Frontend setup:
  - `web/README.md`

## Formal Report Docs

- Documentation index:
  - `docs/README.md`
- Detailed LaTeX report:
  - `docs/report/agentic_multimodal_report.tex`
