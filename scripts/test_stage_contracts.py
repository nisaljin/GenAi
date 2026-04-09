#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def validate_vlm_log(vlm_log: str) -> None:
    if not vlm_log.strip():
        raise ValueError("vlm_log is empty")
    pattern = re.compile(r"^\[(\d+(?:\.\d+)?)s\]\s+.+")
    lines = [line.strip() for line in vlm_log.splitlines() if line.strip()]
    timestamped = [line for line in lines if pattern.match(line)]
    if not timestamped:
        raise ValueError("vlm_log has no timestamped lines like [0.0s] ...")
    bad = [line for line in timestamped if not pattern.match(line)]
    if bad:
        raise ValueError(f"vlm_log has malformed lines: {bad[:3]}")


def validate_events(events: list[dict]) -> None:
    if not events:
        raise ValueError("events list is empty")
    for i, e in enumerate(events):
        if not isinstance(e, dict):
            raise ValueError(f"event[{i}] is not an object")
        if "timestamp_sec" not in e or "original_prompt" not in e:
            raise ValueError(f"event[{i}] missing required fields")
        ts = float(e["timestamp_sec"])
        if ts < 0:
            raise ValueError(f"event[{i}] has negative timestamp")
        prompt = str(e["original_prompt"]).strip()
        if not prompt:
            raise ValueError(f"event[{i}] has empty prompt")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate handoff contracts between stage outputs")
    parser.add_argument("--vlm-log-path", required=True)
    parser.add_argument("--events-path", required=True)
    parser.add_argument("--audio-path", default=None)
    args = parser.parse_args()

    vlm_log_path = Path(args.vlm_log_path)
    events_path = Path(args.events_path)
    if not vlm_log_path.exists():
        raise FileNotFoundError(vlm_log_path)
    if not events_path.exists():
        raise FileNotFoundError(events_path)

    vlm_log = vlm_log_path.read_text(encoding="utf-8")
    validate_vlm_log(vlm_log)

    events = json.loads(events_path.read_text(encoding="utf-8"))
    if isinstance(events, dict) and "data" in events:
        events = events["data"]
    validate_events(events)

    if args.audio_path:
        audio_path = Path(args.audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(audio_path)

    print(
        json.dumps(
            {
                "vlm_log_lines": len([x for x in vlm_log.splitlines() if x.strip()]),
                "num_events": len(events),
                "audio_checked": bool(args.audio_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
