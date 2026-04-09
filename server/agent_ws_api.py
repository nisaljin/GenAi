#!/usr/bin/env python3
import asyncio
import mimetypes
import os
import threading
import time
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in os.sys.path:
    os.sys.path.insert(0, ROOT_DIR)

from main import FoleyOrchestrator  # noqa: E402


app = FastAPI(title="Foley Agent WebSocket API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

WS_HOST = os.getenv("AGENT_WS_HOST", "0.0.0.0")
WS_PORT = int(os.getenv("AGENT_WS_PORT", "8010"))
PUBLIC_BASE_URL = os.getenv("AGENT_PUBLIC_BASE_URL", f"http://localhost:{WS_PORT}")
UPLOAD_DIR = os.path.join(ROOT_DIR, "generated_outputs", "uploads")
VIDEO_OUTPUT_DIR = os.path.join(ROOT_DIR, "generated_outputs", "video")
AUDIO_OUTPUT_DIR = os.path.join(ROOT_DIR, "generated_outputs", "audio")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)
os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)


def _safe_name(name: str) -> str:
    return os.path.basename(name).replace("..", "").strip()


def _artifact_url(kind: str, path: str) -> str:
    filename = _safe_name(path)
    return f"{PUBLIC_BASE_URL}/artifacts/{kind}/{filename}"


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True}


@app.put("/upload-video")
async def upload_video(request: Request, filename: str | None = None) -> dict[str, Any]:
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="Request body is empty.")

    requested = _safe_name(filename or "")
    if not requested:
        requested = f"uploaded_video_{int(time.time())}.mp4"

    root, ext = os.path.splitext(requested)
    if not ext:
        ext = ".mp4"
    saved_name = f"{root}_{int(time.time())}{ext}"
    saved_path = os.path.join(UPLOAD_DIR, saved_name)

    with open(saved_path, "wb") as fh:
        fh.write(body)

    return {"video_path": saved_path, "filename": saved_name}


@app.get("/artifacts/{kind}/{filename}")
def artifact(kind: str, filename: str):
    safe_kind = kind.strip().lower()
    safe_filename = _safe_name(filename)
    if safe_kind == "video":
        base_dir = VIDEO_OUTPUT_DIR
    elif safe_kind == "audio":
        base_dir = AUDIO_OUTPUT_DIR
    else:
        raise HTTPException(status_code=404, detail="Unknown artifact kind.")

    target = os.path.join(base_dir, safe_filename)
    if not os.path.exists(target):
        raise HTTPException(status_code=404, detail="Artifact not found.")

    media_type = mimetypes.guess_type(target)[0] or "application/octet-stream"
    return FileResponse(target, media_type=media_type, filename=safe_filename)


@app.websocket("/ws/foley")
async def websocket_foley(websocket: WebSocket):
    await websocket.accept()
    try:
        init_msg = await websocket.receive_json()
        action = str(init_msg.get("action", "start")).strip().lower()
        if action != "start":
            await websocket.send_json({
                "type": "error",
                "payload": {"message": "First message must be {'action': 'start', ...}"},
            })
            await websocket.close(code=1003)
            return

        video_path = str(init_msg.get("video_path", "")).strip()
        output_path = str(init_msg.get("output_path", "")).strip()
        prompt = str(init_msg.get("prompt", "")).strip()

        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def event_callback(event: dict[str, Any]) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, event)

        orchestrator = FoleyOrchestrator(event_callback=event_callback)

        def run_pipeline() -> None:
            try:
                requested_output = output_path
                if video_path:
                    if not requested_output:
                        requested_output = f"foley_video_{int(time.time())}.mp4"
                    orchestrator.run_pipeline(video_path, requested_output, prompt=prompt)
                else:
                    orchestrator.run_audio_only(prompt=prompt, output_audio_path=requested_output)
            except Exception as e:
                loop.call_soon_threadsafe(
                    queue.put_nowait,
                    {
                        "type": "run_failed",
                        "payload": {"error": str(e)},
                    },
                )

        worker = threading.Thread(target=run_pipeline, daemon=True)
        worker.start()

        while True:
            event = await queue.get()
            if event.get("type") == "run_completed":
                payload = dict(event.get("payload", {}))
                video_path_done = str(payload.get("output_video_path", "")).strip()
                audio_path_done = str(payload.get("output_audio_path", "")).strip()
                if video_path_done:
                    payload["output_url"] = _artifact_url("video", video_path_done)
                if audio_path_done:
                    payload["output_url"] = _artifact_url("audio", audio_path_done)
                event["payload"] = payload
            await websocket.send_json(event)
            event_type = str(event.get("type", ""))
            if event_type in {"run_completed", "run_failed"}:
                await websocket.close(code=1000)
                break

    except WebSocketDisconnect:
        return
    except Exception as e:
        try:
            await websocket.send_json({
                "type": "error",
                "payload": {"message": str(e)},
            })
        except Exception:
            pass
        try:
            await websocket.close(code=1011)
        except Exception:
            pass


def main() -> None:
    uvicorn.run(app, host=WS_HOST, port=WS_PORT, reload=False)


if __name__ == "__main__":
    main()
