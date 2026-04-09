# Stage-by-Stage Pipeline Tests

Run each model stage independently from repo root.

## 1) Perception (video -> `vlm_log`)

```bash
./.venv/bin/python scripts/test_perception_step.py \
  --video-path input_video.mp4 \
  --out-dir stage_outputs/perception
```

Outputs:
- `stage_outputs/perception/vlm_log.txt`
- `stage_outputs/perception/keyframes_and_captions.json`
- `stage_outputs/perception/summary.json`

## 2) Planner (`vlm_log` -> event plan)

```bash
./.venv/bin/python scripts/test_planner_step.py \
  --vlm-log-path stage_outputs/perception/vlm_log.txt \
  --out-dir stage_outputs/planner
```

Outputs:
- `stage_outputs/planner/planner_attempts.json` (raw model outputs by attempt)
- `stage_outputs/planner/events.json`
- `stage_outputs/planner/summary.json`

## 3) Contract Check (`vlm_log` + events)

```bash
./.venv/bin/python scripts/test_stage_contracts.py \
  --vlm-log-path stage_outputs/perception/vlm_log.txt \
  --events-path stage_outputs/planner/events.json
```

## 4) Generation (prompt -> wav)

Use a prompt from `events.json`:

```bash
./.venv/bin/python scripts/test_generation_step.py \
  --prompt "metal clink with short room reverb" \
  --duration 2.0 \
  --out-dir stage_outputs/generation
```

Outputs:
- `stage_outputs/generation/event_0.00.wav`
- `stage_outputs/generation/summary.json`

## 5) Verification (prompt + wav -> CLAP score)

```bash
./.venv/bin/python scripts/test_verification_step.py \
  --prompt "metal clink with short room reverb" \
  --audio-path stage_outputs/generation/event_0.00.wav \
  --out-dir stage_outputs/verification
```

Outputs:
- `stage_outputs/verification/summary.json`

## Notes

- These scripts include compatibility fallbacks for:
  - AudioLDM2 language model mismatch (`GPT2Model` vs generation-capable LM)
  - CLAP processor API change (`audio=` vs `audios=`)
- If a step fails, inspect the JSON summary/attempt files in that step's output directory.
