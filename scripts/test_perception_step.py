#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    Blip2ForConditionalGeneration,
    Blip2Processor,
    LlavaForConditionalGeneration,
)

from stage_utils import dtype_for, ensure_dir, pick_device, write_json, write_text


def extract_keyframes(video_path: str, threshold: float) -> List[Dict[str, Any]]:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames: List[Dict[str, Any]] = []

    success, prev_frame = cap.read()
    if not success:
        cap.release()
        return frames

    frames.append({"timestamp": 0.0, "frame": prev_frame})
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    idx = 1
    while True:
        success, curr_frame = cap.read()
        if not success:
            break
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        mean_diff = np.mean(cv2.absdiff(curr_gray, prev_gray))
        if mean_diff > threshold:
            frames.append({"timestamp": round(idx / fps, 2), "frame": curr_frame})
            prev_gray = curr_gray
        idx += 1

    cap.release()
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Test perception stage (video -> vlm_log)")
    parser.add_argument("--video-path", default="input_video.mp4")
    parser.add_argument("--backend", choices=["llava", "blip2"], default="blip2")
    parser.add_argument("--model", default="Salesforce/blip-image-captioning-base")
    parser.add_argument("--question", default="What sound-relevant events and materials are implied by this scene?")
    parser.add_argument("--cache-dir", default=".hf-cache")
    parser.add_argument("--device", default=None)
    parser.add_argument("--keyframe-threshold", type=float, default=15.0)
    parser.add_argument("--max-keyframes", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--out-dir", default="stage_outputs/perception")
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    device = args.device or pick_device(prefer_gpu=True)
    print(f"[perception] device={device} model={args.model}")

    frames = extract_keyframes(args.video_path, args.keyframe_threshold)
    if not frames:
        raise RuntimeError("No frames extracted")
    first_frame = frames[0]
    if args.max_keyframes > 0 and len(frames) > args.max_keyframes:
        # Keep chronology while reducing compute pressure.
        step = max(1, len(frames) // args.max_keyframes)
        frames = frames[::step][: args.max_keyframes]
        if frames[0]["timestamp"] != 0.0:
            frames.insert(0, first_frame)

    if args.backend == "llava":
        processor = AutoProcessor.from_pretrained(args.model, cache_dir=args.cache_dir)
        model = LlavaForConditionalGeneration.from_pretrained(
            args.model,
            torch_dtype=dtype_for(device),
            cache_dir=args.cache_dir,
        ).to(device)
    else:
        processor = Blip2Processor.from_pretrained(args.model, cache_dir=args.cache_dir)
        model = Blip2ForConditionalGeneration.from_pretrained(
            args.model,
            dtype=dtype_for(device),
            cache_dir=args.cache_dir,
        ).to(device)

    rows: List[Dict[str, Any]] = []
    echo_failures = 0
    for frame_info in frames:
        ts = float(frame_info["timestamp"])
        bgr = frame_info["frame"]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        if args.backend == "llava":
            image = Image.fromarray(rgb)
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": args.question},
                        {"type": "image"},
                    ],
                }
            ]
            prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = processor(text=prompt, images=image, return_tensors="pt")
            inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(dtype=dtype_for(device))

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    num_beams=2,
                )

            input_len = inputs["input_ids"].shape[-1]
            caption = processor.decode(generated_ids[0][input_len:], skip_special_tokens=True).strip()
        else:
            prompt = f"Question: {args.question} Answer:"
            inputs = processor(images=rgb, text=prompt, return_tensors="pt")
            inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    num_beams=3,
                )

            caption = processor.decode(generated_ids[0], skip_special_tokens=True).strip()
        lowered = caption.lower()
        looks_like_echo = (
            lowered == prompt.lower()
            or "describe only sound-relevant" in lowered
            or lowered in {"answer:", "question:", args.question.lower()}
            or re.fullmatch(r"(question:.*answer:?)", lowered) is not None
        )
        if looks_like_echo:
            echo_failures += 1
            if args.backend == "blip2":
                # BLIP2-specific fallback: image-only generation.
                fb_inputs = processor(images=rgb, return_tensors="pt")
                fb_inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in fb_inputs.items()}
                with torch.no_grad():
                    fb_ids = model.generate(
                        **fb_inputs,
                        max_new_tokens=max(32, min(args.max_new_tokens, 48)),
                        do_sample=False,
                        num_beams=3,
                    )
                fb_caption = processor.decode(fb_ids[0], skip_special_tokens=True).strip()
                if fb_caption:
                    caption = fb_caption

        rows.append({"timestamp_sec": ts, "caption": caption})
        print(f"[perception] frame t={ts:.2f}s processed")

    vlm_log = "\n".join(f"[{r['timestamp_sec']}s] {r['caption']}" for r in rows)

    write_json(out_dir / "keyframes_and_captions.json", rows)
    write_text(out_dir / "vlm_log.txt", vlm_log)

    summary = {
        "video_path": str(Path(args.video_path).resolve()),
        "device": device,
        "backend": args.backend,
        "model": args.model,
        "num_keyframes": len(frames),
        "num_lines": len(rows),
        "prompt_echo_failures": echo_failures,
        "vlm_log_path": str((out_dir / "vlm_log.txt").resolve()),
    }
    write_json(out_dir / "summary.json", summary)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
