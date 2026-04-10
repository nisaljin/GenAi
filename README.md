# GenAI Music In Video (Agentic Foley)

Agentic pipeline that generates Foley audio from video (or prompt-only), scores quality with CLAP, and iteratively retries/refines until acceptance criteria are met.

## Repo Structure

- `main.py`: core orchestration and agentic loop.
- `server/`: backend APIs (`agent_ws_api.py`, `multi_model_api.py`).
- `web/`: Next.js frontend for upload + live event streaming.
- `notebooks/`: teammate-facing deep-dive + stage test notebook.
- `scripts/`: stage tests and helper scripts.

## Quick Start

1. Start model API:
```bash
./server/run_multi_model_api.sh
```

2. Start WebSocket bridge:
```bash
./server/run_agent_ws_api.sh
```

3. Start frontend:
```bash
cd web
npm run dev
```

4. Open:
`http://localhost:3000`

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
