# GenAI Music In Video (Agentic Foley)

End-to-end system for generating Foley audio from video with an agentic retry loop.

The pipeline can run in two modes:
- `video` mode: analyze visual frames, plan timed sound events, iteratively synthesize/score, then mux into output video.
- `audio_only` mode: run the same inner loop from prompt text and output an audio file.

## 1. Architecture

### 1.1 Components

- `web/` (Next.js UI)
  - Uploads video
  - Opens WebSocket to stream run events live
  - Renders agent reasoning timeline and inline output players

- `server/agent_ws_api.py` (WebSocket bridge)
  - `PUT /upload-video`
  - `GET /artifacts/{kind}/{filename}`
  - `WS /ws/foley`
  - Runs orchestration in worker thread and forwards events to UI

- `main.py` (`FoleyOrchestrator`)
  - Controls the agentic loop
  - Runs perception/planning/execution/verification steps
  - Stitches final media and writes run trace JSON

- `server/multi_model_api.py` (model serving)
  - `/perception`: VLM analysis from keyframes
  - `/planner`: event planning JSON generation
  - `/execution`: audio synthesis
  - `/verification`: CLAP similarity scoring

### 1.2 Runtime Sequence

1. UI uploads video (`PUT /upload-video`)
2. UI opens WebSocket (`/ws/foley`) and sends start payload
3. WS bridge starts orchestrator pipeline
4. Orchestrator emits events during each stage
5. WS bridge returns events in real time to UI
6. Final artifact URL is attached to `run_completed`
7. UI renders inline `<audio>` or `<video>` player

## 2. Repository Layout

- `main.py` - orchestration and event emission
- `server/` - backend services and run scripts
- `web/` - frontend app
- `notebooks/` - system docs and experimentation notebooks
- `scripts/` - stage tests and utilities
- `generated_outputs/` - runtime outputs (audio/video/uploads/agent_logs)

## 3. Local Setup

### 3.1 Prerequisites

- Python 3.12+
- Node.js 18+
- ffmpeg (recommended for robust muxing)
- GPU optional but strongly recommended for model-serving performance

### 3.2 Python Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r server/requirements.txt
pip install -r server/requirements_multi_model_api.txt
```

### 3.3 Frontend

```bash
cd web
npm install
```

## 4. Run the System

Open 3 terminals.

### Terminal A: Multi-model API

```bash
./server/run_multi_model_api.sh
```

Expected health:

```bash
curl http://localhost:8000/health
```

### Terminal B: WebSocket bridge

```bash
./server/run_agent_ws_api.sh
```

Expected health:

```bash
curl http://localhost:8010/health
```

### Terminal C: Frontend

```bash
cd web
npm run dev
```

Open `http://localhost:3000`.

## 5. Environment Variables

Core vars (typically in `.env`):

- `VLM_API_URL` - base URL for perception endpoint target
- `AUDIO_API_URL` - base URL for execution/verification targets
- `VLM_MODEL_NAME` - VLM identity used by orchestrator prompt metadata
- `MAX_PERCEPTION_FRAMES` - upper bound for sampled keyframes
- `PERCEPTION_RESIZE_TO` - square resize dimension for frame encoding
- `PERCEPTION_CENTER_CROP` - whether to center-crop frames
- `MAX_VIDEO_SECONDS` - max analyzed/muxed duration before trim
- `PROMPT_ONLY_DURATION_SEC` - default audio-only duration

Frontend vars (`web/.env.local`):

- `NEXT_PUBLIC_AGENT_WS_URL` (default `ws://localhost:8010/ws/foley`)
- `NEXT_PUBLIC_AGENT_API_URL` (default `http://localhost:8010`)

## 5.1 Model Matrix

Defaults come from `server/multi_model_api.py`:

| Stage | Purpose | Default Model ID | Endpoint | Override Env |
|---|---|---|---|---|
| Perception (VLM) | Analyze keyframes and produce scene/audio-relevant log | `Qwen/Qwen2-VL-2B-Instruct` | `POST /perception` | `PERCEPTION_MODEL` |
| Planner (LLM) | Convert VLM log to timed Foley event plan JSON | `Qwen/Qwen2.5-7B-Instruct` | `POST /planner` | `PLANNER_MODEL` |
| Execution (Audio) | Generate waveform from text prompt | `facebook/audiogen-medium` | `POST /execution` | `EXECUTION_MODEL` |
| Verification (CLAP) | Score text-audio similarity for agent loop decisions | `laion/clap-htsat-fused` | `POST /verification` | `VERIFICATION_MODEL` |

Related runtime vars:
- `VLM_MODEL_NAME` (orchestrator-side metadata/prompt label)
- `OFFLOAD_AFTER_USE` (model memory behavior on multi-model API)

## 6. WebSocket Contract

Client sends first message:

```json
{
  "action": "start",
  "prompt": "optional",
  "video_path": "/absolute/backend/path/or-empty"
}
```

Server emits event envelope:

```json
{
  "type": "event_name",
  "payload": { "...": "..." }
}
```

Common event types:

- lifecycle: `run_started`, `video_prepared`, `planning_completed`, `run_completed`, `run_failed`
- perception: `vlm_keyframes_extracted`, `vlm_keyframes_downsampled`, `vlm_request_started`, `vlm_response_received`, `vlm_request_failed`, `perception_completed`
- loop: `attempt_started`, `clap_scored`, `decision_made`, `event_completed`

## 7. Agentic Loop Summary

For each planned event:

1. Generate audio candidate
2. Score with CLAP
3. Decide action (`ACCEPT`, `RETRY_REWRITE`, `RETRY_BEST`, `STOP_BEST`)
4. Retry until accepted or retries exhausted
5. Select best candidate and continue

After all events, final timeline audio is mixed and muxed into output media.

## 8. Outputs

Generated under `generated_outputs/`:

- `uploads/` - uploaded source files
- `audio/` - generated wav files
- `video/` - final muxed videos
- `agent_logs/` - run trace JSON (`*_agent_trace.json`)

## 9. Troubleshooting

### Upload fails with `TypeError: Failed to fetch`

Usually CORS preflight failure (`OPTIONS ... 405`). Ensure CORS middleware is active in `agent_ws_api.py`.

### WebSocket not connecting

- Verify bridge is running on `:8010`
- Verify WS URL is `ws://.../ws/foley`
- Install websocket support: `uvicorn[standard]`, `websockets`, or `wsproto`

### Timeline too generic

Common causes:
- prompt override path bypasses VLM timeline
- small VLM model output behavior
- too few keyframes or low token budget in perception

## 10. Team Docs

- System deep dive notebook: `notebooks/agentic_foley_system_deep_dive.ipynb`
- Notebook index: `notebooks/README.md`
- Backend details: `server/README.md`
- Frontend details: `web/README.md`
- Stage test notes: `scripts/README_STAGE_TESTS.md`
