# Server README

Backend services for the agentic Foley system.

This folder contains two runtime services plus helper docs/scripts.

## 1. Services

## 1.1 `agent_ws_api.py` (WebSocket bridge)

Responsibilities:
- Upload ingestion: `PUT /upload-video`
- Artifact serving: `GET /artifacts/{kind}/{filename}`
- Run orchestration over socket: `WS /ws/foley`
- Streams orchestration events to frontend in real time

Behavior notes:
- Starts one worker thread per WS run request
- Converts final output paths into artifact URLs on `run_completed`
- Includes CORS middleware to allow browser uploads from frontend origin

## 1.2 `multi_model_api.py` (Model-serving API)

Endpoints:
- `POST /perception`
- `POST /planner`
- `POST /execution`
- `POST /verification`
- `GET /health`
- `POST /warmup`

Model loading:
- Uses lazy loading with optional unload-after-use (`OFFLOAD_AFTER_USE` behavior)
- Supports cache directory and HF token-based model pulls

## 1.3 `inference_service.py`

Additional/alternate inference logic utilities for model calls and testing flows.

## 1.4 Model Matrix

Defaults are defined in `multi_model_api.py`:

| Stage | Purpose | Default Model ID | Endpoint | Override Env |
|---|---|---|---|---|
| Perception (VLM) | Read image sequence and produce scene/audio timeline text | `Qwen/Qwen2-VL-2B-Instruct` | `POST /perception` | `PERCEPTION_MODEL` |
| Planner (LLM) | Produce structured event plan from VLM log | `Qwen/Qwen2.5-7B-Instruct` | `POST /planner` | `PLANNER_MODEL` |
| Execution (Audio) | Synthesize candidate Foley clips | `facebook/audiogen-medium` | `POST /execution` | `EXECUTION_MODEL` |
| Verification (CLAP) | Compute text-audio similarity score | `laion/clap-htsat-fused` | `POST /verification` | `VERIFICATION_MODEL` |

Additional model/runtime controls:
- `OFFLOAD_AFTER_USE`: unload model after each request to reduce memory pressure.
- `HF_TOKEN` / `HUGGINGFACE_HUB_TOKEN`: access gated/private models.
- `HF_HOME` / `--cache-dir`: local model cache location.

## 2. Startup

Run model API first, then WS bridge.

### 2.1 Multi-model API

```bash
./run_multi_model_api.sh
```

Health:

```bash
curl http://localhost:8000/health
```

### 2.2 WebSocket bridge

```bash
./run_agent_ws_api.sh
```

Health:

```bash
curl http://localhost:8010/health
```

## 3. Request Contracts

## 3.1 `PUT /upload-video?filename=<name>`

- Body: raw file bytes
- Returns:

```json
{
  "video_path": "/absolute/path/to/saved/file",
  "filename": "saved_name.mp4"
}
```

## 3.2 `WS /ws/foley`

First message must be:

```json
{
  "action": "start",
  "prompt": "optional",
  "video_path": "optional absolute path"
}
```

Streams event messages:

```json
{
  "type": "event_name",
  "payload": { "...": "..." }
}
```

Terminal events:
- `run_completed`
- `run_failed`

## 4. Event Coverage

Lifecycle:
- `run_started`
- `video_prepared`
- `planning_completed`
- `run_completed`
- `run_failed`

Perception/VLM:
- `vlm_keyframes_extracted`
- `vlm_keyframes_downsampled`
- `vlm_request_started`
- `vlm_response_received`
- `vlm_request_failed`
- `perception_completed`

Loop internals:
- `attempt_started`
- `clap_scored`
- `decision_made`
- `event_completed`

## 5. Operational Tips

### 5.1 CORS and uploads

If frontend shows upload `TypeError: Failed to fetch` and logs show `OPTIONS ... 405`, CORS/preflight is failing.

### 5.2 WebSocket upgrade warnings

If logs show unsupported upgrade or no websocket library:

```bash
pip install "uvicorn[standard]" websockets wsproto
```

### 5.3 Port conflicts

Default ports:
- `8000` model API
- `8010` WS bridge

If bind fails, check existing listeners and update env vars/script defaults.

### 5.4 Artifact playback issues

`206 Partial Content` logs on artifact requests are normal for browser media seeking/streaming.

## 6. Files and Scripts

- `run_agent_ws_api.sh` - launch WS bridge
- `run_multi_model_api.sh` - launch model-serving stack
- `run_multi_model_api.md`, `run.md` - run notes
- `requirements*.txt` - python deps

## 7. Development Guidance

When changing event payloads:
- keep frontend mapper in sync (`web/components/ui/bolt-style-chat.jsx`)
- avoid silent breaking changes to event field names
- update docs/notebooks with new schema
