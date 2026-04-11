import cv2
import os
import json
import base64
import shutil
import subprocess
import traceback
import time
import re
import numpy as np
import requests
from pydub import AudioSegment
from moviepy import VideoFileClip, AudioFileClip
from typing import List, Dict, Any, Callable, Optional, Set
from dataclasses import dataclass
from dotenv import load_dotenv
from groq import Groq

# Load environment variables from .env file
load_dotenv()


def build_endpoint(base_url: str, preferred_path: str, fallback_path: str = "") -> tuple[str, str]:
    base = base_url.rstrip("/")
    if base.endswith(preferred_path):
        fallback = (base[: -len(preferred_path)] + fallback_path) if fallback_path else ""
        return base, fallback
    if fallback_path and base.endswith(fallback_path):
        preferred = base[: -len(fallback_path)] + preferred_path
        return preferred, base
    preferred = f"{base}{preferred_path}"
    fallback = f"{base}{fallback_path}" if fallback_path else ""
    return preferred, fallback

# ==========================================
# Data Structures
# ==========================================

@dataclass
class AudioEvent:
    timestamp_sec: float
    duration_sec: float # NEW
    original_prompt: str
    refined_prompt: str
    audio_file_path: str = None
    similarity_score: float = 0.0
    agent_trace: List[Dict[str, Any]] = None


@dataclass
class AttemptRecord:
    attempt: int
    prompt: str
    score: float
    raw_final_score: float
    audio_file_path: str
    verifier: Dict[str, Any] = None
    cross_modal: Dict[str, Any] = None
    uncertainty_reasons: List[str] = None
    acceptance_blocked_by_uncertainty: bool = False


@dataclass
class EventAgentState:
    timestamp_sec: float
    duration_sec: float
    best_score: float = -1.0
    best_audio_file_path: str = ""
    best_prompt: str = ""
    attempts: List[AttemptRecord] = None
    reasoning_trace: List[Dict[str, Any]] = None
    accepted: bool = False

    def __post_init__(self):
        if self.attempts is None:
            self.attempts = []
        if self.reasoning_trace is None:
            self.reasoning_trace = []

# ==========================================
# Node 1: Perception (Vision)
# ==========================================

class PerceptionNode:
    def __init__(self, event_emitter: Optional[Callable[[str, Dict[str, Any]], None]] = None):
        self.model_name = os.getenv("VLM_MODEL_NAME", "llava")
        raw_url = os.getenv("VLM_API_URL")
        self.api_url, self.api_url_fallback = build_endpoint(raw_url, "/perception", "/api/generate")
        self.max_frames = int(os.getenv("MAX_PERCEPTION_FRAMES", "12"))
        self.center_crop = os.getenv("PERCEPTION_CENTER_CROP", "1").strip().lower() in {"1", "true", "yes", "on"}
        self.resize_to = int(os.getenv("PERCEPTION_RESIZE_TO", "896"))
        self.event_emitter = event_emitter

    def emit(self, event_type: str, payload: Dict[str, Any]) -> None:
        if not self.event_emitter:
            return
        try:
            self.event_emitter(event_type, payload)
        except Exception:
            pass

    def extract_keyframes(self, video_path: str, threshold: float = 15.0) -> List[Dict[str, Any]]:
        vidcap = cv2.VideoCapture(video_path)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        
        frames = []
        success, prev_frame = vidcap.read()
        
        if not success:
            return frames
            
        frames.append({"timestamp": 0.0, "frame": prev_frame})
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        
        count = 1
        while True:
            success, curr_frame = vidcap.read()
            if not success:
                break
                
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            frame_diff = cv2.absdiff(curr_gray, prev_gray)
            mean_diff = np.mean(frame_diff)
            
            if mean_diff > threshold:
                timestamp = count / fps
                frames.append({"timestamp": round(timestamp, 2), "frame": curr_frame})
                prev_gray = curr_gray 
            
            count += 1
            
        vidcap.release()
        print(f"[Perception Node] Extracted {len(frames)} pertinent keyframes.")
        self.emit("vlm_keyframes_extracted", {
            "keyframe_count": len(frames),
            "threshold": threshold,
        })
        return frames

    def preprocess_frame(self, image: np.ndarray) -> np.ndarray:
        processed = image
        if self.center_crop:
            h, w = processed.shape[:2]
            side = min(h, w)
            top = (h - side) // 2
            left = (w - side) // 2
            processed = processed[top:top + side, left:left + side]

        if self.resize_to > 0:
            processed = cv2.resize(
                processed,
                (self.resize_to, self.resize_to),
                interpolation=cv2.INTER_AREA,
            )
        return processed

    def encode_image_to_base64(self, image: np.ndarray) -> str:
        processed = self.preprocess_frame(image)
        _, buffer = cv2.imencode('.jpg', processed)
        return base64.b64encode(buffer).decode('utf-8')

    def analyze_video(self, video_path: str) -> str:
        keyframes = self.extract_keyframes(video_path, threshold=15.0)
        if len(keyframes) > self.max_frames:
            step = max(len(keyframes) // self.max_frames, 1)
            sampled = keyframes[::step][:self.max_frames]
            print(
                f"[Perception Node] Downsampling keyframes {len(keyframes)} -> {len(sampled)} "
                f"(MAX_PERCEPTION_FRAMES={self.max_frames})."
            )
            self.emit("vlm_keyframes_downsampled", {
                "before": len(keyframes),
                "after": len(sampled),
                "max_frames": self.max_frames,
            })
            keyframes = sampled
        
        if not keyframes:
            return "Error: No frames could be extracted from the video."

        timeline_end = round(float(keyframes[-1]["timestamp"]), 2) if keyframes else 0.0
        system_prompt = (
            "You are an expert Foley artist and sound designer. "
            "You will receive sampled frames from a silent video with known timestamps. "
            "Produce a detailed SOUND timeline with explicit time ranges and acoustic cues.\n\n"
            "STRICT FORMAT RULES:\n"
            "1) Return plain text only (no markdown, no JSON).\n"
            "2) One cue per line.\n"
            "3) Every line MUST start with [start-end] in seconds, e.g. [5.0-8.0].\n"
            "4) Include at least one continuous background/bed ambience cue spanning most of the clip.\n"
            "5) Include discrete foreground cues when visible actions change.\n"
            "6) Use acoustic language (texture, material, intensity, distance, motion).\n"
            f"7) Keep all ranges inside [0.0-{timeline_end}s].\n\n"
            "EXAMPLE STYLE:\n"
            "[0.0-15.0] Forest bed ambience: light wind through leaves, distant soft rustle.\n"
            "[2.5-4.0] Footstep crunch on dry twigs near camera, medium intensity.\n"
            "[5.0-8.0] Bird chirps from upper-left canopy, intermittent and bright.\n"
            "[9.0-12.0] Branch swish as subject passes foliage, close and airy."
        )

        images_base64 = []
        timestamp_context = "Timestamps for the provided frames:\n"
        
        for idx, kf in enumerate(keyframes):
            b64_img = self.encode_image_to_base64(kf["frame"])
            images_base64.append(b64_img)
            timestamp_context += f"Image {idx+1}: [{kf['timestamp']}s]\n"

        full_prompt = (
            f"{system_prompt}\n\n"
            f"{timestamp_context}\n"
            "Now generate the timeline using the strict format."
        )
        frame_timestamps = [kf["timestamp"] for kf in keyframes]

        print(f"[Perception Node] Sending {len(images_base64)} frames to VLM ({self.model_name})...")
        self.emit("vlm_request_started", {
            "model": self.model_name,
            "frame_count": len(images_base64),
            "frame_timestamps": frame_timestamps,
            "endpoint": self.api_url,
            "fallback_endpoint": self.api_url_fallback,
        })

        payload_multi = {
            "images_base64": images_base64,
            "prompt": full_prompt,
        }
        payload_legacy = {
            "model": self.model_name,
            "prompt": full_prompt,
            "images": images_base64,
            "stream": False
        }

        try:
            response = requests.post(self.api_url, json=payload_multi)
            if response.status_code == 404 and self.api_url_fallback:
                response = requests.post(self.api_url_fallback, json=payload_legacy)
            if not response.ok:
                raise RuntimeError(f"Perception API error {response.status_code}: {response.text}")
            body = response.json()
            vlm_log = body.get("vlm_log") or body.get("response", "")
            
            print(f"[Perception Node] Successfully generated VLM Log.")
            self.emit("vlm_response_received", {
                "model": self.model_name,
                "log_length": len(vlm_log),
                "preview": vlm_log[:500],
            })
            return vlm_log
            
        except Exception as e:
            print(f"[Perception Node] API Error: {e}")
            self.emit("vlm_request_failed", {
                "error": str(e),
            })
            return "Error: Failed to reach VLM."

# ==========================================
# Node 2: Planner (The Brain)
# ==========================================

class PlannerNode:
    def __init__(self, model_name="llama-3.3-70b-versatile"):
        self.model_name = model_name
        # Groq will automatically pick up GROQ_API_KEY from the environment
        self.client = Groq() 

    def create_audio_plan(self, vlm_log: str, video_duration_sec: float) -> List[AudioEvent]:
        duration = max(float(video_duration_sec), 0.5)
        prompt = (
            "You are a Foley Audio Director. Read this video log and output a JSON object containing a 'data' array of acoustic events. "
            "Strictly use this format: {\"data\": [{\"timestamp_sec\": 0.0, \"duration_sec\": 2.0, \"original_prompt\": \"...\"}]}. "
            f"The video duration is {duration:.2f} seconds. "
            "CRITICAL: All events MUST fit inside [0, video_duration]. "
            "For every event: timestamp_sec >= 0, duration_sec > 0, and timestamp_sec + duration_sec <= video_duration. "
            "CRITICAL: If the scene has a continuous atmosphere (like a room or nature), create ONE long event that spans nearly the whole duration. "
            "The 'original_prompt' MUST describe the AUDIO, not the visual. Use acoustic keywords (e.g., 'Resonant, wet splash, ambient wind'). "
            "Do not output markdown, only raw JSON."
            f"\n\nVideo Log:\n{vlm_log}"
        )
        
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model_name,
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            raw_json = chat_completion.choices[0].message.content
            events_data = json.loads(raw_json).get("data", [])
            
            events: List[AudioEvent] = []
            for e in events_data:
                ts = max(float(e.get("timestamp_sec", 0.0)), 0.0)
                dur = max(float(e.get("duration_sec", 2.0)), 0.5)
                if ts >= duration:
                    continue
                dur = min(dur, duration - ts)
                if dur <= 0:
                    continue

                events.append(AudioEvent(
                    timestamp_sec=ts,
                    duration_sec=dur,
                    original_prompt=str(e.get("original_prompt", "")).strip(),
                    refined_prompt=""
                ))

            if not events:
                # Conservative fallback: one ambient event across full clip.
                events = [AudioEvent(
                    timestamp_sec=0.0,
                    duration_sec=duration,
                    original_prompt="continuous ambient room tone matching visible scene",
                    refined_prompt=""
                )]
            
            print(f"[Planner Node] Generated {len(events)} prompts.")
            return events
        except Exception as e:
            print(f"[Planner Node] Error parsing JSON: {e}")
            return []

    def refine_prompt(self, failed_prompt: str, score: float) -> str:
        prompt = f"The audio prompt '{failed_prompt}' scored {score:.2f}/1.0 in acoustic similarity. Rewrite it to be more distinct, highly detailed, and descriptive for an AI audio generator."
        
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model_name,
                temperature=0.7
            )
            refined = chat_completion.choices[0].message.content.strip()
            print(f"[Planner Node] Refined prompt: '{refined}'")
            return refined
        except Exception as e:
            print(f"[Planner Node] Error refining: {e}")
            return failed_prompt + ", highly detailed, distinct sound"

    def decide_iteration(
        self,
        event: AudioEvent,
        current_prompt: str,
        score: float,
        threshold: float,
        attempt: int,
        max_retries: int,
        state: EventAgentState,
    ) -> Dict[str, Any]:
        attempts_summary = [
            {
                "attempt": a.attempt,
                "prompt": a.prompt,
                "score": round(a.score, 4),
            }
            for a in state.attempts
        ]
        payload = {
            "event": {
                "timestamp_sec": round(event.timestamp_sec, 3),
                "duration_sec": round(event.duration_sec, 3),
                "original_prompt": event.original_prompt,
            },
            "current_prompt": current_prompt,
            "score": round(score, 4),
            "threshold": threshold,
            "attempt": attempt,
            "max_retries": max_retries,
            "best_score_so_far": round(state.best_score, 4),
            "best_prompt_so_far": state.best_prompt,
            "attempts_so_far": attempts_summary,
        }
        controller_prompt = (
            "You are an agent controller for iterative Foley generation. "
            "Choose one action based on current CLAP score and history. "
            "Allowed actions: ACCEPT, RETRY_REWRITE, RETRY_BEST, STOP_BEST. "
            "Return ONLY JSON with keys: action, reasoning, confidence, next_prompt. "
            "Rules: "
            "1) If score >= threshold, prefer ACCEPT. "
            "2) If final attempt and score < threshold, use STOP_BEST. "
            "3) For RETRY_REWRITE, provide an improved next_prompt. "
            "4) For RETRY_BEST, next_prompt should be best_prompt_so_far if available. "
            "5) confidence in [0,1]."
            f"\n\nState:\n{json.dumps(payload)}"
        )
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": controller_prompt}],
                model=self.model_name,
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            raw = chat_completion.choices[0].message.content
            decision = json.loads(raw)
            action = str(decision.get("action", "")).strip().upper()
            if action not in {"ACCEPT", "RETRY_REWRITE", "RETRY_BEST", "STOP_BEST"}:
                action = "RETRY_REWRITE"

            reasoning = str(decision.get("reasoning", "")).strip()
            next_prompt_raw = decision.get("next_prompt", "")
            next_prompt = next_prompt_raw.strip() if isinstance(next_prompt_raw, str) else ""
            confidence = float(decision.get("confidence", 0.5))
            confidence = min(max(confidence, 0.0), 1.0)

            return {
                "action": action,
                "reasoning": reasoning or "No reasoning provided by controller.",
                "confidence": confidence,
                "next_prompt": next_prompt,
                "source": "groq",
            }
        except Exception as e:
            print(f"[Planner Node] Controller decision error: {e}")
            if score >= threshold:
                action = "ACCEPT"
            elif attempt >= max_retries:
                action = "STOP_BEST"
            elif state.best_prompt and len(state.attempts) >= 2 and state.attempts[-1].score < state.attempts[-2].score:
                action = "RETRY_BEST"
            else:
                action = "RETRY_REWRITE"

            return {
                "action": action,
                "reasoning": "Heuristic fallback controller decision.",
                "confidence": 0.4,
                "next_prompt": "",
                "source": "heuristic",
            }

# ==========================================
# Node 3: Execution (Synthesis)
# ==========================================

class ExecutionNode:
    def __init__(self):
        base_url = os.getenv("AUDIO_API_URL").rstrip('/')
        self.api_url, self.api_url_fallback = build_endpoint(base_url, "/execution", "/generate")
        self.output_dir = os.getenv("GENERATED_AUDIO_DIR", "./generated_outputs/audio")
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_audio(self, prompt: str, timestamp: float, duration: float, attempt: int) -> str:
        safe_duration = max(float(duration), 0.5)
        print(f"[Execution Node] Requesting {safe_duration}s audio for: '{prompt[:40]}...'")
        
        response = requests.post(self.api_url, json={"prompt": prompt, "duration": safe_duration})
        if response.status_code == 404 and self.api_url_fallback:
            response = requests.post(self.api_url_fallback, json={"prompt": prompt, "duration": safe_duration})
        if not response.ok:
            raise RuntimeError(f"Execution API error {response.status_code}: {response.text}")
        
        b64_audio = response.json().get("audio_base64")
        file_path = f"{self.output_dir}/event_{timestamp}_v{attempt}.wav"
        
        with open(file_path, "wb") as fh:
            fh.write(base64.b64decode(b64_audio))
            
        print(f"[Execution Node] Synthesized -> {file_path}")
        return file_path

# ==========================================
# Node 4: Verification (CLAP Evaluator)
# ==========================================

class VerificationNode:
    def __init__(self, threshold: float = 0.50): # Lowered threshold slightly for real models
        self.threshold = threshold
        base_url = os.getenv("AUDIO_API_URL").rstrip('/')
        self.api_url, self.api_url_fallback = build_endpoint(base_url, "/verification", "/evaluate")

    def evaluate(self, prompt: str, audio_path: str) -> Dict[str, Any]:
        with open(audio_path, "rb") as fh:
            b64_audio = base64.b64encode(fh.read()).decode('utf-8')
            
        response = requests.post(self.api_url, json={"prompt": prompt, "audio_base64": b64_audio})
        if response.status_code == 404 and self.api_url_fallback:
            response = requests.post(self.api_url_fallback, json={"prompt": prompt, "audio_base64": b64_audio})
        if not response.ok:
            raise RuntimeError(f"Verification API error {response.status_code}: {response.text}")
        
        body = response.json()
        score_primary = float(body.get("score_primary", body.get("similarity_score", 0.0)))
        score_secondary = float(body.get("score_secondary", score_primary))
        final_score = float(body.get("final_score", body.get("similarity_score", score_primary)))
        score_gap = float(body.get("score_gap", abs(score_primary - score_secondary)))
        agreement_ok = bool(body.get("agreement_ok", True))
        verifier_gap_delta = float(body.get("verifier_gap_delta", 0.25))
        print(
            "[Verification Node] "
            f"primary={score_primary:.2f} secondary={score_secondary:.2f} "
            f"final={final_score:.2f} gap={score_gap:.2f} agreement_ok={agreement_ok}"
        )
        return {
            "score_primary": score_primary,
            "score_secondary": score_secondary,
            "final_score": final_score,
            "score_gap": score_gap,
            "agreement_ok": agreement_ok,
            "verifier_gap_delta": verifier_gap_delta,
        }

# ==========================================
# Final Stage: Stitching & Orchestration
# ==========================================

class FoleyOrchestrator:
    def __init__(self, event_callback: Optional[Callable[[Dict[str, Any]], None]] = None):
        self.perception = PerceptionNode(event_emitter=self.emit_event)
        self.planner = PlannerNode()
        self.execution = ExecutionNode()
        self.verification = VerificationNode(threshold=0.50)
        self.max_retries = 3
        self.clap_score_min = float(os.getenv("CLAP_SCORE_MIN", "0.0"))
        self.clap_score_max = float(os.getenv("CLAP_SCORE_MAX", "10.0"))
        self.quality_threshold = float(os.getenv("QUALITY_THRESHOLD", "0.60"))
        self.self_consistency_runs = max(1, int(os.getenv("SELF_CONSISTENCY_RUNS", "3")))
        self.cross_modal_threshold = float(os.getenv("CROSS_MODAL_AGREEMENT_THRESHOLD", "0.35"))
        self.prompt_only_verifier_abs_gap_max = float(os.getenv("PROMPT_ONLY_VERIFIER_ABS_GAP_MAX", "8.0"))
        self.prompt_only_verifier_rel_gap_max = float(os.getenv("PROMPT_ONLY_VERIFIER_REL_GAP_MAX", "0.75"))
        self.max_video_seconds = float(os.getenv("MAX_VIDEO_SECONDS", "15"))
        self.video_output_dir = os.getenv("GENERATED_VIDEO_DIR", "./generated_outputs/video")
        self.temp_output_dir = os.getenv("GENERATED_TEMP_DIR", "./generated_outputs/tmp")
        self.agent_log_dir = os.getenv("AGENT_LOG_DIR", "./generated_outputs/agent_logs")
        self.event_callback = event_callback
        os.makedirs(self.video_output_dir, exist_ok=True)
        os.makedirs(self.temp_output_dir, exist_ok=True)
        os.makedirs(self.agent_log_dir, exist_ok=True)

    def emit_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        if not self.event_callback:
            return
        try:
            self.event_callback({"type": event_type, "payload": payload})
        except Exception as e:
            print(f"[Orchestrator] Event callback error: {e}")

    @staticmethod
    def _prompt_tokens(text: str) -> Set[str]:
        stop = {
            "the", "a", "an", "and", "or", "for", "with", "from", "into", "onto", "near",
            "continuous", "ambient", "background", "scene", "appropriate", "sound", "audio",
            "detailed", "highly", "quality", "matching", "visible",
        }
        tokens = {
            tok for tok in re.findall(r"[a-zA-Z]+", text.lower())
            if len(tok) > 2 and tok not in stop
        }
        return tokens

    @staticmethod
    def _clip01(value: float) -> float:
        return min(max(value, 0.0), 1.0)

    def normalize_quality_score(self, raw_final_score: float) -> float:
        denom = self.clap_score_max - self.clap_score_min
        if denom <= 0:
            return 0.0
        normalized = (raw_final_score - self.clap_score_min) / denom
        return self._clip01(normalized)

    def verifier_agreement_ok(self, verifier_result: Dict[str, Any], prompt_only_mode: bool = False) -> bool:
        raw_agreement_ok = bool(verifier_result.get("agreement_ok", True))
        if raw_agreement_ok:
            return True
        if not prompt_only_mode:
            return False

        score_primary = float(verifier_result.get("score_primary", 0.0))
        score_secondary = float(verifier_result.get("score_secondary", 0.0))
        score_gap = float(verifier_result.get("score_gap", abs(score_primary - score_secondary)))
        rel_gap = score_gap / max(abs(score_primary), abs(score_secondary), 1e-6)
        abs_ok = score_gap <= self.prompt_only_verifier_abs_gap_max
        rel_ok = rel_gap <= self.prompt_only_verifier_rel_gap_max
        return abs_ok or rel_ok

    def extract_expected_audio_keywords(self, vlm_log: str) -> Set[str]:
        candidate_words = {
            "footstep", "footsteps", "gravel", "wind", "water", "splash", "rain", "thunder",
            "bird", "birds", "chirp", "rustle", "leaves", "door", "metal", "wood", "glass",
            "engine", "car", "vehicle", "crowd", "applause", "fire", "crackle", "wave", "ocean",
            "river", "bike", "chain", "bell", "dog", "bark", "cat", "meow", "click", "swish",
        }
        tokens = self._prompt_tokens(vlm_log)
        matched = {t for t in tokens if t in candidate_words}
        if not matched:
            matched = set(list(tokens)[:6])
        return matched

    def compute_cross_modal_agreement(self, prompt: str, expected_keywords: Set[str]) -> Dict[str, Any]:
        if not expected_keywords:
            return {
                "agreement_score": 1.0,
                "matched_keywords": [],
                "missing_keywords": [],
                "agreement_ok": True,
            }
        prompt_tokens = self._prompt_tokens(prompt)
        matched = sorted([w for w in expected_keywords if w in prompt_tokens])
        missing = sorted([w for w in expected_keywords if w not in prompt_tokens])
        agreement_score = len(matched) / max(len(expected_keywords), 1)
        agreement_ok = agreement_score >= self.cross_modal_threshold
        return {
            "agreement_score": agreement_score,
            "matched_keywords": matched,
            "missing_keywords": missing,
            "agreement_ok": agreement_ok,
        }

    def evaluate_planner_self_consistency(self, vlm_log: str, video_duration_sec: float) -> Dict[str, Any]:
        plans: List[List[AudioEvent]] = []
        for _ in range(self.self_consistency_runs):
            plans.append(self.planner.create_audio_plan(vlm_log, video_duration_sec))

        non_empty = [p for p in plans if p]
        if not non_empty:
            return {
                "stable": False,
                "num_runs": self.self_consistency_runs,
                "count_variance": 0.0,
                "avg_timestamp_diff": 0.0,
                "avg_prompt_jaccard": 0.0,
                "selected_plan": [],
            }

        counts = [len(p) for p in non_empty]
        count_variance = float(np.var(counts)) if len(counts) > 1 else 0.0

        base = sorted(non_empty[0], key=lambda e: e.timestamp_sec)
        ts_diffs: List[float] = []
        jaccards: List[float] = []
        for other in non_empty[1:]:
            other_sorted = sorted(other, key=lambda e: e.timestamp_sec)
            m = min(len(base), len(other_sorted))
            if m == 0:
                continue
            for i in range(m):
                ts_diffs.append(abs(base[i].timestamp_sec - other_sorted[i].timestamp_sec))
                a = self._prompt_tokens(base[i].original_prompt)
                b = self._prompt_tokens(other_sorted[i].original_prompt)
                union = a | b
                inter = a & b
                jaccards.append((len(inter) / len(union)) if union else 1.0)

        avg_timestamp_diff = float(np.mean(ts_diffs)) if ts_diffs else 0.0
        avg_prompt_jaccard = float(np.mean(jaccards)) if jaccards else 1.0

        stable = (
            count_variance <= 1.0
            and avg_timestamp_diff <= 1.5
            and avg_prompt_jaccard >= 0.2
        )

        median_count = int(np.median(counts))
        selected = min(
            non_empty,
            key=lambda p: (abs(len(p) - median_count), len(p)),
        )

        return {
            "stable": stable,
            "num_runs": self.self_consistency_runs,
            "count_variance": count_variance,
            "avg_timestamp_diff": avg_timestamp_diff,
            "avg_prompt_jaccard": avg_prompt_jaccard,
            "selected_plan": selected,
        }

    def run_event_agent(
        self,
        event: AudioEvent,
        expected_keywords: Optional[Set[str]] = None,
        prompt_only_mode: bool = False,
    ) -> AudioEvent:
        state = EventAgentState(timestamp_sec=event.timestamp_sec, duration_sec=event.duration_sec)
        current_prompt = event.original_prompt

        for attempt in range(1, self.max_retries + 1):
            self.emit_event("attempt_started", {
                "timestamp_sec": event.timestamp_sec,
                "duration_sec": event.duration_sec,
                "attempt": attempt,
                "max_retries": self.max_retries,
                "prompt": current_prompt,
            })
            print(
                f"  -> [Agent] Event {event.timestamp_sec:.2f}s | "
                f"attempt={attempt}/{self.max_retries}"
            )

            audio_path = self.execution.generate_audio(
                current_prompt,
                event.timestamp_sec,
                event.duration_sec,
                attempt,
            )
            verifier_result = self.verification.evaluate(current_prompt, audio_path)
            raw_final_score = float(verifier_result.get("final_score", 0.0))
            score = self.normalize_quality_score(raw_final_score)
            verifier_agreement_ok = self.verifier_agreement_ok(
                verifier_result,
                prompt_only_mode=prompt_only_mode,
            )
            cross_modal = self.compute_cross_modal_agreement(
                current_prompt,
                expected_keywords or set(),
            )
            self.emit_event("verifier_scored", {
                "timestamp_sec": event.timestamp_sec,
                "attempt": attempt,
                "prompt": current_prompt,
                "score_primary": round(verifier_result["score_primary"], 4),
                "score_secondary": round(verifier_result["score_secondary"], 4),
                "score_gap": round(verifier_result["score_gap"], 4),
                "agreement_ok": verifier_agreement_ok,
                "agreement_ok_raw": verifier_result["agreement_ok"],
                "raw_final_score": round(raw_final_score, 4),
                "quality_score_normalized": round(score, 4),
                "quality_threshold": round(self.quality_threshold, 4),
                "verifier_gap_delta": verifier_result["verifier_gap_delta"],
                "audio_file_path": audio_path,
                "threshold": self.quality_threshold,
            })
            self.emit_event("cross_modal_checked", {
                "timestamp_sec": event.timestamp_sec,
                "attempt": attempt,
                "prompt": current_prompt,
                "agreement_score": round(cross_modal["agreement_score"], 4),
                "agreement_ok": cross_modal["agreement_ok"],
                "matched_keywords": cross_modal["matched_keywords"],
                "missing_keywords": cross_modal["missing_keywords"][:8],
                "threshold": self.cross_modal_threshold,
            })
            # Keep old event for compatibility with existing dashboards.
            self.emit_event("clap_scored", {
                "timestamp_sec": event.timestamp_sec,
                "attempt": attempt,
                "prompt": current_prompt,
                "score": round(score, 4),
                "raw_final_score": round(raw_final_score, 4),
                "audio_file_path": audio_path,
                "threshold": self.quality_threshold,
            })

            if score > state.best_score:
                state.best_score = score
                state.best_audio_file_path = audio_path
                state.best_prompt = current_prompt

            uncertainty_reasons = []
            if not verifier_agreement_ok:
                uncertainty_reasons.append("verifier_disagreement")
            if not cross_modal.get("agreement_ok", True):
                uncertainty_reasons.append("cross_modal_mismatch")
            uncertain = len(uncertainty_reasons) > 0
            acceptance_blocked_by_uncertainty = False

            state.attempts.append(AttemptRecord(
                attempt=attempt,
                prompt=current_prompt,
                score=score,
                raw_final_score=raw_final_score,
                audio_file_path=audio_path,
                verifier=verifier_result,
                cross_modal=cross_modal,
                uncertainty_reasons=uncertainty_reasons,
                acceptance_blocked_by_uncertainty=False,
            ))

            decision = self.planner.decide_iteration(
                event=event,
                current_prompt=current_prompt,
                score=score,
                threshold=self.quality_threshold,
                attempt=attempt,
                max_retries=self.max_retries,
                state=state,
            )
            if decision["action"] == "ACCEPT" and uncertain:
                acceptance_blocked_by_uncertainty = True
                if attempt >= self.max_retries:
                    decision["action"] = "STOP_BEST"
                    decision["reasoning"] = (
                        f"{decision['reasoning']} | ACCEPT blocked due to uncertainty "
                        f"({', '.join(uncertainty_reasons)}); forcing STOP_BEST."
                    )
                else:
                    decision["action"] = "RETRY_REWRITE"
                    decision["reasoning"] = (
                        f"{decision['reasoning']} | ACCEPT blocked due to uncertainty "
                        f"({', '.join(uncertainty_reasons)})."
                    )
                state.attempts[-1].acceptance_blocked_by_uncertainty = True

            state.reasoning_trace.append({
                "attempt": attempt,
                "action": decision["action"],
                "reasoning": decision["reasoning"],
                "confidence": decision["confidence"],
                "controller_source": decision["source"],
                "current_prompt": current_prompt,
                "score": round(score, 4),
                "raw_final_score": round(raw_final_score, 4),
                "normalized_score": round(score, 4),
                "best_score_so_far": round(state.best_score, 4),
                "verifier": verifier_result,
                "cross_modal": cross_modal,
                "uncertainty_reasons": uncertainty_reasons,
                "acceptance_blocked_by_uncertainty": acceptance_blocked_by_uncertainty,
            })
            print(
                f"  -> [Agent Decision] action={decision['action']} "
                f"source={decision['source']} confidence={decision['confidence']:.2f}"
            )
            print(f"  -> [Agent Reasoning] {decision['reasoning']}")
            self.emit_event("decision_made", {
                "timestamp_sec": event.timestamp_sec,
                "attempt": attempt,
                "action": decision["action"],
                "reasoning": decision["reasoning"],
                "confidence": decision["confidence"],
                "controller_source": decision["source"],
                "current_prompt": current_prompt,
                "score": round(score, 4),
                "raw_final_score": round(raw_final_score, 4),
                "normalized_score": round(score, 4),
                "quality_threshold": round(self.quality_threshold, 4),
                "best_score_so_far": round(state.best_score, 4),
                "acceptance_blocked_by_uncertainty": acceptance_blocked_by_uncertainty,
            })
            if uncertain:
                self.emit_event("uncertainty_flagged", {
                    "timestamp_sec": event.timestamp_sec,
                    "attempt": attempt,
                    "reasons": uncertainty_reasons,
                    "score_gap": round(verifier_result.get("score_gap", 0.0), 4),
                    "cross_modal_score": round(cross_modal.get("agreement_score", 0.0), 4),
                    "raw_final_score": round(raw_final_score, 4),
                    "quality_score_normalized": round(score, 4),
                    "quality_threshold": round(self.quality_threshold, 4),
                    "acceptance_blocked_by_uncertainty": acceptance_blocked_by_uncertainty,
                })

            if decision["action"] == "ACCEPT":
                print("  -> [Agent] Accepting candidate for this event.")
                state.accepted = True
                break

            if decision["action"] == "STOP_BEST":
                print("  -> [Agent] Stopping retries and using best candidate so far.")
                break

            next_prompt_raw = decision.get("next_prompt", "")
            next_prompt = next_prompt_raw.strip() if isinstance(next_prompt_raw, str) else ""
            if decision["action"] == "RETRY_BEST" and state.best_prompt:
                current_prompt = state.best_prompt
            elif next_prompt:
                current_prompt = next_prompt
            else:
                current_prompt = self.planner.refine_prompt(current_prompt, score)

            print("  -> [Agent] Candidate rejected. Re-planning prompt.")

        event.audio_file_path = state.best_audio_file_path
        event.similarity_score = max(state.best_score, 0.0)
        event.refined_prompt = state.best_prompt or event.original_prompt
        event.agent_trace = state.reasoning_trace
        self.emit_event("event_completed", {
            "timestamp_sec": event.timestamp_sec,
            "duration_sec": event.duration_sec,
            "selected_prompt": event.refined_prompt,
            "final_score": round(event.similarity_score, 4),
            "quality_threshold": round(self.quality_threshold, 4),
            "audio_file_path": event.audio_file_path,
            "accepted": state.accepted,
        })
        if not state.accepted:
            print(
                f"  -> [Agent] Final fallback for event at {event.timestamp_sec:.2f}s: "
                f"best_score={state.best_score:.2f}"
            )
        return event

    def build_run_report(
        self,
        input_video_path: str,
        output_video_path: str,
        events: List[AudioEvent],
        vlm_log: str,
    ) -> Dict[str, Any]:
        return {
            "input_video_path": input_video_path,
            "output_video_path": output_video_path,
            "quality_threshold": self.quality_threshold,
            "clap_score_min": self.clap_score_min,
            "clap_score_max": self.clap_score_max,
            "event_count": len(events),
            "vlm_log": vlm_log,
            "events": [
                {
                    "timestamp_sec": e.timestamp_sec,
                    "duration_sec": e.duration_sec,
                    "original_prompt": e.original_prompt,
                    "selected_prompt": e.refined_prompt,
                    "final_score": e.similarity_score,
                    "audio_file_path": e.audio_file_path,
                    "attempts": [
                        {
                            "attempt": i + 1,
                            "action": t.get("action"),
                            "score": t.get("score"),
                            "raw_final_score": t.get("raw_final_score"),
                            "normalized_score": t.get("normalized_score"),
                            "reasoning": t.get("reasoning"),
                            "confidence": t.get("confidence"),
                            "controller_source": t.get("controller_source"),
                            "prompt": t.get("current_prompt"),
                            "uncertainty_reasons": t.get("uncertainty_reasons"),
                            "acceptance_blocked_by_uncertainty": t.get("acceptance_blocked_by_uncertainty"),
                            "verifier": t.get("verifier"),
                            "cross_modal": t.get("cross_modal"),
                        }
                        for i, t in enumerate(e.agent_trace or [])
                    ],
                }
                for e in events
            ],
        }

    def save_run_report(self, output_video_path: str, report: Dict[str, Any]) -> str:
        base_name = os.path.splitext(os.path.basename(output_video_path))[0]
        report_path = os.path.join(self.agent_log_dir, f"{base_name}_agent_trace.json")
        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"[Agent] Run trace saved to {report_path}")
        return report_path

    def prepare_video(self, video_path: str) -> tuple[str, float, bool]:
        clip = VideoFileClip(video_path)
        duration = float(clip.duration)
        if duration <= self.max_video_seconds:
            clip.close()
            return video_path, duration, False

        trimmed_path = os.path.join(self.temp_output_dir, "trimmed_input.mp4")
        print(
            f"[Orchestrator] Trimming input video from {duration:.2f}s to "
            f"{self.max_video_seconds:.2f}s."
        )
        trimmed = clip.subclipped(0, self.max_video_seconds)
        trimmed.write_videofile(trimmed_path, codec="libx264", audio_codec="aac")
        trimmed.close()
        clip.close()
        return trimmed_path, self.max_video_seconds, True

    def stitch_audio_to_video(self, video_path: str, events: List[AudioEvent], output_path: str):
        print("[Stitcher] Assembling final video...")
        
        video_clip = VideoFileClip(video_path)
        base_audio = AudioSegment.silent(duration=int(video_clip.duration * 1000))

        for event in events:
            if event.audio_file_path and os.path.exists(event.audio_file_path):
                foley_clip = AudioSegment.from_wav(event.audio_file_path)
                insert_position_ms = int(event.timestamp_sec * 1000)
                base_audio = base_audio.overlay(foley_clip, position=insert_position_ms)

        temp_audio_path = os.path.join(self.temp_output_dir, "mixed_audio.wav")
        base_audio.export(temp_audio_path, format="wav")
        video_clip.close()

        # ffmpeg muxing is more robust than moviepy audio attachment across codec/container variations.
        if shutil.which("ffmpeg"):
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                video_path,
                "-i",
                temp_audio_path,
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-shortest",
                output_path,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg mux failed: {result.stderr.strip()}")
        else:
            # Fallback path for environments without ffmpeg binary available.
            vclip = VideoFileClip(video_path)
            aclip = AudioFileClip(temp_audio_path)
            final_video = vclip.with_audio(aclip)
            final_video.write_videofile(output_path, codec="libx264", audio_codec="aac")
            final_video.close()
            aclip.close()
            vclip.close()
        
        os.remove(temp_audio_path)
        print(f"[Stitcher] Success! Final video saved to {output_path}")

    def run_pipeline(self, video_path: str, output_path: str, prompt: str = ""):
        print(f"--- Starting Autonomous Foley Generation for {video_path} ---")
        self.emit_event("run_started", {
            "mode": "video",
            "video_path": video_path,
            "output_path": output_path,
            "prompt": prompt,
            "max_retries": self.max_retries,
            "quality_threshold": self.quality_threshold,
        })
        if not output_path:
            output_path = f"foley_video_{int(time.time())}.mp4"
        if not os.path.isabs(output_path) and os.path.dirname(output_path) == "":
            output_path = os.path.join(self.video_output_dir, output_path)
        prepared_video_path, prepared_duration, is_temp = self.prepare_video(video_path)
        self.emit_event("video_prepared", {
            "prepared_video_path": prepared_video_path,
            "duration_sec": round(prepared_duration, 3),
            "was_trimmed": is_temp,
        })
        print(f"[Orchestrator] Using timeline duration: {prepared_duration:.2f}s")
        try:
            vlm_log = ""
            if prompt.strip():
                audio_plan = [AudioEvent(
                    timestamp_sec=0.0,
                    duration_sec=prepared_duration,
                    original_prompt=prompt.strip(),
                    refined_prompt=""
                )]
                self.emit_event("planning_completed", {
                    "event_count": len(audio_plan),
                    "mode": "prompt_override",
                    "events": [
                        {
                            "timestamp_sec": audio_plan[0].timestamp_sec,
                            "duration_sec": audio_plan[0].duration_sec,
                            "original_prompt": audio_plan[0].original_prompt,
                        }
                    ],
                })
            else:
                vlm_log = self.perception.analyze_video(prepared_video_path)
                if vlm_log.startswith("Error:"):
                    raise RuntimeError(f"Perception failed: {vlm_log}")
                self.emit_event("perception_completed", {"vlm_log": vlm_log})
                sc = self.evaluate_planner_self_consistency(vlm_log, prepared_duration)
                self.emit_event("self_consistency_checked", {
                    "stable": sc["stable"],
                    "num_runs": sc["num_runs"],
                    "count_variance": round(sc["count_variance"], 4),
                    "avg_timestamp_diff": round(sc["avg_timestamp_diff"], 4),
                    "avg_prompt_jaccard": round(sc["avg_prompt_jaccard"], 4),
                })
                audio_plan = sc["selected_plan"]
                if not audio_plan:
                    print("[Agent] Planner returned no events. Injecting one ambient fallback event.")
                    audio_plan = [AudioEvent(
                        timestamp_sec=0.0,
                        duration_sec=prepared_duration,
                        original_prompt="continuous ambient scene-appropriate background texture",
                        refined_prompt=""
                    )]
                self.emit_event("planning_completed", {
                    "event_count": len(audio_plan),
                    "events": [
                        {
                            "timestamp_sec": e.timestamp_sec,
                            "duration_sec": e.duration_sec,
                            "original_prompt": e.original_prompt,
                        }
                        for e in audio_plan
                    ],
                })
                expected_keywords = self.extract_expected_audio_keywords(vlm_log)
            if not prompt.strip():
                self.emit_event("cross_modal_expected_keywords", {
                    "keywords": sorted(list(expected_keywords))[:12],
                })
            else:
                expected_keywords = set()

            final_events = []
            for event in audio_plan:
                final_events.append(self.run_event_agent(event, expected_keywords=expected_keywords))
                print("-" * 40)

            self.stitch_audio_to_video(prepared_video_path, final_events, output_path)
            report = self.build_run_report(
                input_video_path=video_path,
                output_video_path=output_path,
                events=final_events,
                vlm_log=vlm_log,
            )
            report_path = self.save_run_report(output_path, report)
            self.emit_event("run_completed", {
                "mode": "video",
                "output_video_path": output_path,
                "report_path": report_path,
                "event_count": len(final_events),
            })
        except Exception as e:
            err = str(e)
            self.emit_event("run_failed", {
                "error": err,
                "traceback": traceback.format_exc(),
            })
            raise
        finally:
            if is_temp and os.path.exists(prepared_video_path):
                os.remove(prepared_video_path)

    def run_audio_only(self, prompt: str, output_audio_path: str = ""):
        clean_prompt = prompt.strip()
        if not clean_prompt:
            raise ValueError("Prompt is required for audio-only generation.")

        default_duration = float(os.getenv("PROMPT_ONLY_DURATION_SEC", "6"))
        safe_duration = max(default_duration, 0.5)

        self.emit_event("run_started", {
            "mode": "audio_only",
            "prompt": clean_prompt,
            "max_retries": self.max_retries,
            "quality_threshold": self.quality_threshold,
        })
        event = AudioEvent(
            timestamp_sec=0.0,
            duration_sec=safe_duration,
            original_prompt=clean_prompt,
            refined_prompt="",
        )
        self.emit_event("planning_completed", {
            "mode": "audio_only",
            "event_count": 1,
            "events": [
                {
                    "timestamp_sec": 0.0,
                    "duration_sec": safe_duration,
                    "original_prompt": clean_prompt,
                }
            ],
        })

        final_event = self.run_event_agent(event, prompt_only_mode=True)
        if not final_event.audio_file_path or not os.path.exists(final_event.audio_file_path):
            raise RuntimeError("Audio generation failed to produce an output file.")

        if not output_audio_path:
            output_audio_path = os.path.join(
                self.execution.output_dir,
                f"foley_audio_{int(time.time())}.wav",
            )
        elif not os.path.isabs(output_audio_path):
            output_audio_path = os.path.join(self.execution.output_dir, output_audio_path)

        if os.path.abspath(final_event.audio_file_path) != os.path.abspath(output_audio_path):
            shutil.copyfile(final_event.audio_file_path, output_audio_path)

        report = self.build_run_report(
            input_video_path="",
            output_video_path=output_audio_path,
            events=[final_event],
            vlm_log="",
        )
        report_path = self.save_run_report(output_audio_path, report)
        self.emit_event("run_completed", {
            "mode": "audio_only",
            "output_audio_path": output_audio_path,
            "report_path": report_path,
            "event_count": 1,
        })

# ==========================================
# Execution Entry Point
# ==========================================
if __name__ == "__main__":
    orchestrator = FoleyOrchestrator()
    orchestrator.run_pipeline("input_video.mp4", "output_foley_video.mp4")
