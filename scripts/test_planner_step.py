#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from stage_utils import (
    dtype_for,
    ensure_dir,
    extract_first_json,
    pick_device,
    planner_fallback_from_vlm_log,
    write_json,
    write_text,
)


def generate_text(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    device: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    tok = tokenizer(prompt, return_tensors="pt", truncation=True)
    tok = {k: v.to(device) if hasattr(v, "to") else v for k, v in tok.items()}

    with torch.no_grad():
        out = model.generate(
            **tok,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
        )

    return tokenizer.decode(out[0], skip_special_tokens=True).strip()


def events_from_parsed(parsed: Dict[str, Any]) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for e in parsed.get("data", []):
        if not isinstance(e, dict):
            continue
        if "timestamp_sec" not in e or "original_prompt" not in e:
            continue
        try:
            ts = float(e["timestamp_sec"])
        except Exception:
            continue
        dur = float(e.get("duration_sec", 2.0) or 2.0)
        if dur <= 0:
            dur = 2.0
        prompt = str(e["original_prompt"]).strip()
        if not prompt:
            continue
        events.append(
            {
                "timestamp_sec": max(0.0, ts),
                "duration_sec": dur,
                "original_prompt": prompt,
            }
        )
    return sorted(events, key=lambda x: x["timestamp_sec"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Test planner stage (vlm_log -> event plan)")
    parser.add_argument("--vlm-log-path", required=True)
    parser.add_argument("--model", default="google/flan-t5-large")
    parser.add_argument("--cache-dir", default=".hf-cache")
    parser.add_argument("--device", default=None)
    parser.add_argument("--planner-attempts", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--fallback-max-events", type=int, default=12)
    parser.add_argument("--out-dir", default="stage_outputs/planner")
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    raw_vlm_log = Path(args.vlm_log_path).read_text(encoding="utf-8")
    timestamped_lines = [
        line.strip()
        for line in raw_vlm_log.splitlines()
        if re.match(r"^\[\d+(?:\.\d+)?s\]\s+.+", line.strip())
    ]
    vlm_log = "\n".join(timestamped_lines) if timestamped_lines else raw_vlm_log
    if not vlm_log.strip():
        raise ValueError("vlm_log is empty")

    device = args.device or pick_device(prefer_gpu=True)
    print(f"[planner] device={device} model={args.model}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model,
        dtype=dtype_for(device),
        cache_dir=args.cache_dir,
    ).to(device)

    base_prompt = (
        "Return ONLY strict JSON object. No extra words. "
        'Schema: {"data":[{"timestamp_sec":number,"duration_sec":number,"original_prompt":string}]}. '
        "Each event must contain concrete numeric timestamp_sec and duration_sec values. "
        "Use acoustic wording only and chronological ordering. "
        f"Video log:\n{vlm_log}"
    )

    events: List[Dict[str, Any]] = []
    attempts_data: List[Dict[str, str]] = []

    for attempt in range(1, args.planner_attempts + 1):
        prompt = base_prompt
        if attempt > 1:
            prompt += (
                "\nPrevious output was invalid. "
                "Do not output schema templates like timestamp:number. "
                'Output one JSON object only with top-level key "data" and real values.'
            )

        raw = generate_text(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        attempts_data.append({"attempt": str(attempt), "raw": raw})

        try:
            parsed = extract_first_json(raw)
            events = events_from_parsed(parsed)
            if events:
                break
        except Exception as exc:
            attempts_data[-1]["parse_error"] = str(exc)

    if not events:
        events = planner_fallback_from_vlm_log(vlm_log, max_events=args.fallback_max_events)

    write_text(out_dir / "vlm_log_input.txt", vlm_log)
    write_json(out_dir / "planner_attempts.json", attempts_data)
    write_json(out_dir / "events.json", events)

    summary = {
        "device": device,
        "model": args.model,
        "num_attempts": len(attempts_data),
        "num_events": len(events),
        "events_path": str((out_dir / "events.json").resolve()),
    }
    write_json(out_dir / "summary.json", summary)

    print(json.dumps(summary, indent=2))
    if not events:
        raise RuntimeError("Planner produced no events")


if __name__ == "__main__":
    main()
