#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.io.wavfile
from diffusers import AudioLDM2Pipeline

from stage_utils import (
    dtype_for,
    ensure_audioldm2_language_model,
    ensure_dir,
    pick_device,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test generation stage (prompt -> wav)")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--timestamp", type=float, default=0.0)
    parser.add_argument("--model", default="cvssp/audioldm2")
    parser.add_argument("--cache-dir", default=".hf-cache")
    parser.add_argument("--device", default=None)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--out-dir", default="stage_outputs/generation")
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    device = args.device or pick_device(prefer_gpu=True)
    print(f"[generation] device={device} model={args.model}")

    pipe = AudioLDM2Pipeline.from_pretrained(
        args.model,
        torch_dtype=dtype_for(device),
        cache_dir=args.cache_dir,
    )
    pipe = ensure_audioldm2_language_model(
        pipe=pipe,
        model_name=args.model,
        device=device,
        cache_dir=args.cache_dir,
    )
    pipe.to(device)

    enhanced_prompt = (
        f"{args.prompt}, high quality Foley sound effect, cinematic, crisp, detailed, 48kHz"
    )
    negative_prompt = (
        "music, speech, human voice, background noise, static, muffled, noisy, electronic, digital"
    )

    audio = pipe(
        prompt=enhanced_prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        audio_length_in_s=max(0.1, args.duration),
    ).audios[0]

    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val
    audio_int16 = (audio * 32767).astype(np.int16)

    out_path = out_dir / f"event_{args.timestamp:.2f}.wav"
    scipy.io.wavfile.write(str(out_path), rate=args.sample_rate, data=audio_int16)

    summary = {
        "device": device,
        "model": args.model,
        "prompt": args.prompt,
        "duration_sec": float(args.duration),
        "audio_path": str(out_path.resolve()),
    }
    write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
