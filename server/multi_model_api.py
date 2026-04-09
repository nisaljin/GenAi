#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import gc
import io
import json
import os
import time
import traceback
from dataclasses import dataclass
from typing import Any

import librosa
import numpy as np
import scipy.io.wavfile
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from PIL import Image
from huggingface_hub import login as hf_login
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    ClapModel,
    ClapProcessor,
    pipeline,
)

try:
    from audiocraft.models import AudioGen
except Exception:  # pragma: no cover
    AudioGen = None


DEFAULT_VLM = "Qwen/Qwen2-VL-2B-Instruct"
DEFAULT_PLANNER = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_EXECUTION = "facebook/audiogen-medium"
DEFAULT_VERIFIER = "laion/clap-htsat-fused"

WAV_SAMPLE_RATE = 16000
CLAP_SAMPLE_RATE = 48000


def _log(message: str) -> None:
    print(message, flush=True)


class PerceptionRequest(BaseModel):
    images_base64: list[str] = Field(..., description="List of JPG/PNG images as base64 strings")
    prompt: str = Field(
        default="Describe all sound-relevant events in chronological order with timestamps.",
        description="Instruction for visual analysis",
    )
    max_new_tokens: int = 256


class PlannerRequest(BaseModel):
    vlm_log: str
    system_prompt: str | None = None
    temperature: float = 0.2
    max_new_tokens: int = 384


class ExecutionRequest(BaseModel):
    prompt: str
    duration: float = 5.0


class VerificationRequest(BaseModel):
    prompt: str
    audio_base64: str


@dataclass
class RuntimeConfig:
    cache_dir: str | None
    hf_token: str | None
    perception_model: str
    planner_model: str
    execution_model: str
    verification_model: str
    device: str
    dtype: torch.dtype
    offload_after_use: bool


class ModelRegistry:
    def __init__(self, config: RuntimeConfig):
        self.config = config
        self._perception_pipe = None
        self._planner_tokenizer = None
        self._planner_model = None
        self._execution_model = None
        self._verification_model = None
        self._verification_processor = None

    def _torch_dtype(self) -> torch.dtype:
        return self.config.dtype if self.config.device == "cuda" else torch.float32

    def load_perception(self) -> Any:
        if self._perception_pipe is not None:
            _log("[load][perception] using cached pipeline")
            return self._perception_pipe

        _log(f"[load][perception] start model={self.config.perception_model}")
        t0 = time.time()
        self._perception_pipe = pipeline(
            task="image-text-to-text",
            model=self.config.perception_model,
            device_map="auto" if self.config.device == "cuda" else None,
            torch_dtype=self._torch_dtype(),
            token=self.config.hf_token,
            model_kwargs={
                "cache_dir": self.config.cache_dir,
            },
        )
        _log(f"[load][perception] done elapsed_sec={time.time() - t0:.2f}")
        return self._perception_pipe

    def load_planner(self) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
        if self._planner_tokenizer is not None and self._planner_model is not None:
            _log("[load][planner] using cached tokenizer+model")
            return self._planner_tokenizer, self._planner_model

        _log(f"[load][planner] start model={self.config.planner_model}")
        t0 = time.time()
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.planner_model,
            cache_dir=self.config.cache_dir,
            token=self.config.hf_token,
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.config.planner_model,
            torch_dtype=self._torch_dtype(),
            cache_dir=self.config.cache_dir,
            token=self.config.hf_token,
            device_map="auto" if self.config.device == "cuda" else None,
        )

        self._planner_tokenizer = tokenizer
        self._planner_model = model
        _log(f"[load][planner] done elapsed_sec={time.time() - t0:.2f}")
        return tokenizer, model

    def load_execution(self):
        if AudioGen is None:
            raise RuntimeError(
                "audiocraft is not installed. Install with: pip install audiocraft"
            )

        if self._execution_model is not None:
            _log("[load][execution] using cached model")
            return self._execution_model

        _log(f"[load][execution] start model={self.config.execution_model}")
        t0 = time.time()
        model = AudioGen.get_pretrained(
            self.config.execution_model,
            device=self.config.device,
        )
        self._execution_model = model
        _log(f"[load][execution] done elapsed_sec={time.time() - t0:.2f}")
        return model

    def load_verification(self) -> tuple[ClapProcessor, ClapModel]:
        if self._verification_processor is not None and self._verification_model is not None:
            _log("[load][verification] using cached processor+model")
            return self._verification_processor, self._verification_model

        _log(f"[load][verification] start model={self.config.verification_model}")
        t0 = time.time()
        processor = ClapProcessor.from_pretrained(
            self.config.verification_model,
            cache_dir=self.config.cache_dir,
            token=self.config.hf_token,
        )
        model = ClapModel.from_pretrained(
            self.config.verification_model,
            cache_dir=self.config.cache_dir,
            token=self.config.hf_token,
        ).to(self.config.device)

        self._verification_processor = processor
        self._verification_model = model
        _log(f"[load][verification] done elapsed_sec={time.time() - t0:.2f}")
        return processor, model

    def _clear_cuda_cache(self) -> None:
        if self.config.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def unload_perception(self) -> None:
        self._perception_pipe = None
        gc.collect()
        self._clear_cuda_cache()

    def unload_planner(self) -> None:
        self._planner_tokenizer = None
        self._planner_model = None
        gc.collect()
        self._clear_cuda_cache()

    def unload_execution(self) -> None:
        self._execution_model = None
        gc.collect()
        self._clear_cuda_cache()

    def unload_verification(self) -> None:
        self._verification_processor = None
        self._verification_model = None
        gc.collect()
        self._clear_cuda_cache()

    def warmup_all(self) -> dict[str, str]:
        status: dict[str, str] = {}
        for name, loader in (
            ("perception", self.load_perception),
            ("planner", self.load_planner),
            ("execution", self.load_execution),
            ("verification", self.load_verification),
        ):
            _log(f"[warmup][{name}] start")
            t0 = time.time()
            try:
                loader()
                status[name] = "ok"
                _log(f"[warmup][{name}] ok elapsed_sec={time.time() - t0:.2f}")
            except Exception as exc:  # pragma: no cover
                status[name] = f"error: {exc}"
                _log(f"[warmup][{name}] error elapsed_sec={time.time() - t0:.2f} detail={exc}")
        return status


def decode_b64_image(image_b64: str) -> Image.Image:
    image_bytes = base64.b64decode(image_b64)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def decode_b64_audio(audio_b64: str) -> np.ndarray:
    audio_bytes = base64.b64decode(audio_b64)
    audio_array, _ = librosa.load(io.BytesIO(audio_bytes), sr=CLAP_SAMPLE_RATE)
    return audio_array


def build_app(registry: ModelRegistry) -> FastAPI:
    app = FastAPI(title="Multi-Model Foley API")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "ok": True,
            "device": registry.config.device,
            "models": {
                "perception": registry.config.perception_model,
                "planner": registry.config.planner_model,
                "execution": registry.config.execution_model,
                "verification": registry.config.verification_model,
            },
        }

    @app.post("/warmup")
    def warmup() -> dict[str, Any]:
        return {"status": registry.warmup_all()}

    @app.post("/perception")
    def perception(req: PerceptionRequest) -> dict[str, Any]:
        try:
            if not req.images_base64:
                raise HTTPException(status_code=400, detail="images_base64 cannot be empty")

            imgs = [decode_b64_image(i) for i in req.images_base64]
            vlm = registry.load_perception()

            messages = [
                {
                    "role": "user",
                    "content": [
                        *[{"type": "image", "image": img} for img in imgs],
                        {"type": "text", "text": req.prompt},
                    ],
                }
            ]

            out = vlm(
                text=messages,
                max_new_tokens=req.max_new_tokens,
                return_full_text=False,
            )

            text = ""
            if isinstance(out, list) and out:
                first = out[0]
                if isinstance(first, dict):
                    text = str(first.get("generated_text", "")).strip()
                else:
                    text = str(first).strip()
            return {"vlm_log": text}
        except HTTPException:
            raise
        except Exception as exc:
            _log(f"[error][perception] {exc}\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        finally:
            if registry.config.offload_after_use:
                _log("[offload][perception] unloading model from memory")
                registry.unload_perception()

    @app.post("/planner")
    def planner(req: PlannerRequest) -> dict[str, Any]:
        try:
            tokenizer, model = registry.load_planner()

            system_prompt = req.system_prompt or (
                "You are a Foley planning assistant. Output ONLY valid JSON with key 'data'. "
                "Each element must be {timestamp_sec, duration_sec, original_prompt}."
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": req.vlm_log},
            ]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            model_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            generated = model.generate(
                **model_inputs,
                do_sample=req.temperature > 0,
                temperature=max(req.temperature, 1e-6),
                max_new_tokens=req.max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )

            new_tokens = generated[0][model_inputs["input_ids"].shape[-1] :]
            raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            parsed: dict[str, Any]
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = {"raw": raw}

            return {"raw": raw, "parsed": parsed}
        except Exception as exc:
            _log(f"[error][planner] {exc}\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        finally:
            if registry.config.offload_after_use:
                _log("[offload][planner] unloading model from memory")
                registry.unload_planner()

    @app.post("/execution")
    def execution(req: ExecutionRequest) -> dict[str, Any]:
        try:
            if req.duration <= 0:
                raise HTTPException(status_code=400, detail="duration must be > 0")

            model = registry.load_execution()
            model.set_generation_params(duration=float(req.duration))
            wav = model.generate([req.prompt])[0].cpu().numpy()

            max_val = np.max(np.abs(wav))
            if max_val > 0:
                wav = wav / max_val
            wav_i16 = (wav * 32767).astype(np.int16)

            byte_io = io.BytesIO()
            scipy.io.wavfile.write(byte_io, rate=WAV_SAMPLE_RATE, data=wav_i16)
            b64_audio = base64.b64encode(byte_io.getvalue()).decode("utf-8")
            return {"audio_base64": b64_audio, "sample_rate": WAV_SAMPLE_RATE}
        except HTTPException:
            raise
        except Exception as exc:
            _log(f"[error][execution] {exc}\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        finally:
            if registry.config.offload_after_use:
                _log("[offload][execution] unloading model from memory")
                registry.unload_execution()

    @app.post("/verification")
    def verification(req: VerificationRequest) -> dict[str, Any]:
        try:
            processor, model = registry.load_verification()
            audio = decode_b64_audio(req.audio_base64)

            inputs = processor(
                text=[req.prompt],
                audios=audio,
                return_tensors="pt",
                padding=True,
                sampling_rate=CLAP_SAMPLE_RATE,
            )
            inputs = {
                k: (v.to(registry.config.device) if hasattr(v, "to") else v)
                for k, v in inputs.items()
            }

            with torch.no_grad():
                outputs = model(**inputs)
                score = float(outputs.logits_per_audio[0][0].item())

            return {"similarity_score": score}
        except Exception as exc:
            _log(f"[error][verification] {exc}\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        finally:
            if registry.config.offload_after_use:
                _log("[offload][verification] unloading model from memory")
                registry.unload_verification()

    return app


def pick_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve Perception/Planner/Execution/Verification models over FastAPI")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--cache-dir", default=os.getenv("HF_HOME") or ".hf-cache")
    parser.add_argument("--hf-token", default=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN"))
    parser.add_argument("--perception-model", default=os.getenv("PERCEPTION_MODEL", DEFAULT_VLM))
    parser.add_argument("--planner-model", default=os.getenv("PLANNER_MODEL", DEFAULT_PLANNER))
    parser.add_argument("--execution-model", default=os.getenv("EXECUTION_MODEL", DEFAULT_EXECUTION))
    parser.add_argument("--verification-model", default=os.getenv("VERIFICATION_MODEL", DEFAULT_VERIFIER))
    parser.add_argument(
        "--offload-after-use",
        action="store_true",
        default=os.getenv("OFFLOAD_AFTER_USE", "1").strip().lower() in {"1", "true", "yes", "on"},
        help="Unload each stage model after every request to reduce VRAM pressure.",
    )
    parser.add_argument(
        "--no-offload-after-use",
        action="store_false",
        dest="offload_after_use",
        help="Keep models resident in memory between requests.",
    )
    parser.add_argument("--uvicorn-log-level", default=os.getenv("UVICORN_LOG_LEVEL", "info"))
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Download/load all models at startup.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = pick_device()
    dtype = torch.float16 if device == "cuda" else torch.float32

    if args.hf_token:
        hf_login(token=args.hf_token, add_to_git_credential=False)

    config = RuntimeConfig(
        cache_dir=args.cache_dir,
        hf_token=args.hf_token,
        perception_model=args.perception_model,
        planner_model=args.planner_model,
        execution_model=args.execution_model,
        verification_model=args.verification_model,
        device=device,
        dtype=dtype,
        offload_after_use=args.offload_after_use,
    )

    _log("[startup] Multi-model Foley API configuration")
    _log(f"[startup] device={config.device} dtype={config.dtype}")
    _log(f"[startup] cache_dir={config.cache_dir}")
    _log(f"[startup] perception_model={config.perception_model}")
    _log(f"[startup] planner_model={config.planner_model}")
    _log(f"[startup] execution_model={config.execution_model}")
    _log(f"[startup] verification_model={config.verification_model}")
    _log(f"[startup] offload_after_use={config.offload_after_use}")
    _log(f"[startup] uvicorn_log_level={args.uvicorn_log_level}")

    registry = ModelRegistry(config)
    app = build_app(registry)

    if args.warmup:
        _log("[startup] warmup requested")
        status = registry.warmup_all()
        _log(f"[warmup] {status}")

    uvicorn.run(app, host=args.host, port=args.port, log_level=args.uvicorn_log_level)


if __name__ == "__main__":
    main()
