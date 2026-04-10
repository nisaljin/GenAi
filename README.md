# GenAI Music In Video (Agentic Foley)

Agentic pipeline that generates Foley audio from video (or prompt-only), scores quality with CLAP, and iteratively retries/refines until acceptance criteria are met.

## Prerequisites

- Python `3.10+` (tested with Python `3.12`)
- Node.js `18+` and npm
- `ffmpeg` available on `PATH` (required for robust video/audio muxing)

## Repo Structure

- `main.py`: core orchestration and agentic loop.
- `server/`: backend APIs (`agent_ws_api.py`, `multi_model_api.py`).
- `web/`: Next.js frontend for upload + live event streaming.
- `notebooks/`: teammate-facing deep-dive + stage test notebook.
- `scripts/`: stage tests and helper scripts.

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
