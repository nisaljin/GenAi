import argparse
import atexit
import base64
import io
import re
import subprocess
import threading
import time
from typing import Optional

import librosa
import numpy as np
import scipy.io.wavfile
import torch
import uvicorn
from diffusers import AudioLDM2Pipeline
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import ClapModel, ClapProcessor

AUDIOLDM2_MODEL = "cvssp/audioldm2"
CLAP_MODEL = "laion/larger_clap_general"
WAV_SAMPLE_RATE = 16000
EVAL_SAMPLE_RATE = 48000


class AudioRequest(BaseModel):
    prompt: str
    duration: float = 5.0


class EvalRequest(BaseModel):
    prompt: str
    audio_base64: str


class ModelRegistry:
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir
        self.audio_pipe: Optional[AudioLDM2Pipeline] = None
        self.clap_model: Optional[ClapModel] = None
        self.clap_processor: Optional[ClapProcessor] = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

    def load(self) -> None:
        if self.audio_pipe is not None and self.clap_model is not None and self.clap_processor is not None:
            return

        print(f"[startup] Loading models on device={self.device}, dtype={self.dtype}")
        self.audio_pipe = AudioLDM2Pipeline.from_pretrained(
            AUDIOLDM2_MODEL,
            torch_dtype=self.dtype,
            cache_dir=self.cache_dir,
        )
        self.audio_pipe.to(self.device)

        self.clap_model = ClapModel.from_pretrained(
            CLAP_MODEL,
            cache_dir=self.cache_dir,
        ).to(self.device)
        self.clap_processor = ClapProcessor.from_pretrained(
            CLAP_MODEL,
            cache_dir=self.cache_dir,
        )
        print("[startup] Models ready.")


registry = ModelRegistry(cache_dir=None)
app = FastAPI(title="Foley Inference Service")


@app.get("/health")
def health():
    loaded = (
        registry.audio_pipe is not None
        and registry.clap_model is not None
        and registry.clap_processor is not None
    )
    return {
        "ok": True,
        "device": registry.device,
        "models_loaded": loaded,
    }


@app.post("/generate")
def generate_audio(req: AudioRequest):
    try:
        if req.duration <= 0:
            raise HTTPException(status_code=400, detail="duration must be > 0")

        registry.load()

        enhanced_prompt = (
            f"{req.prompt}, high quality Foley sound effect, cinematic, crisp, detailed, 48kHz"
        )
        negative_prompt = (
            "music, speech, human voice, background noise, static, muffled, noisy, electronic, digital"
        )

        audio = registry.audio_pipe(
            prompt=enhanced_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=150,
            guidance_scale=3.5,
            audio_length_in_s=req.duration,
        ).audios[0]

        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        audio_int16 = (audio * 32767).astype(np.int16)

        byte_io = io.BytesIO()
        scipy.io.wavfile.write(byte_io, rate=WAV_SAMPLE_RATE, data=audio_int16)
        b64_audio = base64.b64encode(byte_io.getvalue()).decode("utf-8")
        return {"audio_base64": b64_audio}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/evaluate")
def evaluate_audio(req: EvalRequest):
    try:
        registry.load()

        audio_bytes = base64.b64decode(req.audio_base64)
        audio_array, _ = librosa.load(io.BytesIO(audio_bytes), sr=EVAL_SAMPLE_RATE)

        try:
            inputs = registry.clap_processor(
                text=[req.prompt],
                audio=audio_array,
                return_tensors="pt",
                padding=True,
                sampling_rate=EVAL_SAMPLE_RATE,
            )
        except (TypeError, ValueError):
            inputs = registry.clap_processor(
                text=[req.prompt],
                audios=audio_array,
                return_tensors="pt",
                padding=True,
                sampling_rate=EVAL_SAMPLE_RATE,
            )
        inputs = {k: v.to(registry.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = registry.clap_model(**inputs)
            score = outputs.logits_per_audio.item() / 100.0
        return {"similarity_score": score}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _start_cloudflared_tunnel(port: int) -> Optional[subprocess.Popen]:
    cmd = [
        "cloudflared",
        "tunnel",
        "--url",
        f"http://127.0.0.1:{port}",
        "--no-autoupdate",
    ]
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except FileNotFoundError:
        print("[tunnel] cloudflared binary not found. Install it or run behind a public VM/IP.")
        return None

    def _watch_logs():
        assert proc.stdout is not None
        for line in proc.stdout:
            print(f"[cloudflared] {line.rstrip()}")
            match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", line)
            if match:
                print(f"[tunnel] Public URL: {match.group(0)}")
                break

    t = threading.Thread(target=_watch_logs, daemon=True)
    t.start()
    time.sleep(1.0)
    return proc


def main():
    parser = argparse.ArgumentParser(
        description="Download models and run Foley inference API."
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument(
        "--expose-web",
        action="store_true",
        help="Open a public URL with cloudflared (if installed).",
    )
    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="Skip startup model warm-load/download.",
    )
    args = parser.parse_args()

    global registry
    registry = ModelRegistry(cache_dir=args.cache_dir)

    if not args.skip_warmup:
        registry.load()

    tunnel_proc = None
    if args.expose_web:
        tunnel_proc = _start_cloudflared_tunnel(args.port)

    if tunnel_proc is not None:
        atexit.register(tunnel_proc.terminate)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
