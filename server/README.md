# Server README

This folder contains backend services used by the agentic Foley pipeline.

## Files

- `agent_ws_api.py`
  - WebSocket bridge for live run events.
  - Handles upload endpoint and artifact serving.
  - Entry point for frontend runtime integration.

- `multi_model_api.py`
  - Hosts model endpoints:
    - `/perception`
    - `/planner`
    - `/execution`
    - `/verification`

- `inference_service.py`
  - Alternate inference service implementation/utilities.

## Run

### 1) Multi-model API
```bash
./run_multi_model_api.sh
```

### 2) WebSocket bridge
```bash
./run_agent_ws_api.sh
```

## Health Checks

```bash
curl http://localhost:8000/health
curl http://localhost:8010/health
```

## Notes

- If browser uploads fail with `OPTIONS ... 405`, verify CORS middleware is enabled in `agent_ws_api.py`.
- If WebSocket fails to upgrade, ensure websocket dependencies are installed (`websockets`/`wsproto` or `uvicorn[standard]`).

