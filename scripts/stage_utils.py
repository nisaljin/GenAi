from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from transformers import AutoModelForCausalLM


@dataclass
class StageConfig:
    cache_dir: str = "./.hf-cache"
    keyframe_threshold: float = 15.0
    planner_attempts: int = 4
    planner_max_new_tokens: int = 384


def pick_device(prefer_gpu: bool = True) -> str:
    if prefer_gpu and torch.cuda.is_available():
        return "cuda"
    if prefer_gpu and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def dtype_for(device: str) -> torch.dtype:
    return torch.float16 if device in {"cuda", "mps"} else torch.float32


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_text(path: str | Path, text: str) -> None:
    Path(path).write_text(text, encoding="utf-8")


def write_json(path: str | Path, obj: Any) -> None:
    Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")


def normalize_json_like(text: str) -> str:
    t = text.strip()
    if not t:
        return t

    if t.startswith("```"):
        t = t.strip("`")
        if t.startswith("json"):
            t = t[4:].strip()

    if "{" not in t and "}" not in t and '"data"' in t:
        return "{" + t + "}"

    return t


def extract_first_json(text: str) -> Dict[str, Any]:
    normalized = normalize_json_like(text)

    for pattern in (r"\{.*\}", r"\[.*\]"):
        m = re.search(pattern, normalized, flags=re.DOTALL)
        if not m:
            continue

        candidate = m.group(0)
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            repaired = candidate.replace("'", '"')
            repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
            parsed = json.loads(repaired)

        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list) and parsed and all(isinstance(item, dict) for item in parsed):
            return {"data": parsed}

    raise ValueError(f"No usable JSON object found. raw={text[:400]}")


def planner_fallback_from_vlm_log(vlm_log: str, max_events: int = 12) -> List[Dict[str, Any]]:
    pattern = re.compile(r"^\[(\d+(?:\.\d+)?)s\]\s*(.+?)\s*$")
    events: List[Dict[str, Any]] = []
    seen = set()

    for line in vlm_log.splitlines():
        m = pattern.match(line.strip())
        if not m:
            continue
        ts = float(m.group(1))
        desc = re.sub(r"\s+", " ", m.group(2)).strip()
        if not desc:
            continue
        key = (round(ts, 2), desc.lower())
        if key in seen:
            continue
        seen.add(key)
        events.append(
            {
                "timestamp_sec": max(0.0, ts),
                "duration_sec": 2.0,
                "original_prompt": desc,
            }
        )
        if len(events) >= max_events:
            break

    return sorted(events, key=lambda x: float(x["timestamp_sec"]))


def clap_processor_inputs(processor: Any, prompt: str, audio_array: np.ndarray, sampling_rate: int) -> Dict[str, Any]:
    try:
        return processor(
            text=[prompt],
            audio=audio_array,
            return_tensors="pt",
            padding=True,
            sampling_rate=sampling_rate,
        )
    except (TypeError, ValueError):
        return processor(
            text=[prompt],
            audios=audio_array,
            return_tensors="pt",
            padding=True,
            sampling_rate=sampling_rate,
        )


def ensure_audioldm2_language_model(pipe: Any, model_name: str, device: str, cache_dir: str | None) -> Any:
    lm = getattr(pipe, "language_model", None)
    if lm is not None and not hasattr(lm, "_update_model_kwargs_for_generation"):
        fixed_lm = AutoModelForCausalLM.from_pretrained(
            model_name,
            subfolder="language_model",
            torch_dtype=dtype_for(device),
            cache_dir=cache_dir,
        )
        if hasattr(pipe, "register_modules"):
            pipe.register_modules(language_model=fixed_lm)
        else:
            pipe.language_model = fixed_lm
    return pipe
