#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import librosa
import torch
from transformers import ClapModel, ClapProcessor

from stage_utils import clap_processor_inputs, ensure_dir, pick_device, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Test verification stage (prompt + wav -> clap score)")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--audio-path", required=True)
    parser.add_argument("--model", default="laion/larger_clap_general")
    parser.add_argument("--cache-dir", default=".hf-cache")
    parser.add_argument("--device", default=None)
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument("--out-dir", default="stage_outputs/verification")
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    audio_path = Path(args.audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    device = args.device or pick_device(prefer_gpu=True)
    print(f"[verification] device={device} model={args.model}")

    clap_model = ClapModel.from_pretrained(args.model, cache_dir=args.cache_dir).to(device)
    clap_processor = ClapProcessor.from_pretrained(args.model, cache_dir=args.cache_dir)

    audio_array, _ = librosa.load(str(audio_path), sr=args.sample_rate)
    inputs = clap_processor_inputs(
        processor=clap_processor,
        prompt=args.prompt,
        audio_array=audio_array,
        sampling_rate=args.sample_rate,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = clap_model(**inputs)
        score = float(outputs.logits_per_audio.item() / 100.0)

    summary = {
        "device": device,
        "model": args.model,
        "prompt": args.prompt,
        "audio_path": str(audio_path.resolve()),
        "similarity_score": score,
    }
    write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
