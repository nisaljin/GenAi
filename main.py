import cv2
import os
import json
import base64
import shutil
import subprocess
import numpy as np
import requests
from pydub import AudioSegment
from moviepy import VideoFileClip, AudioFileClip
from typing import List, Dict, Any
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

# ==========================================
# Node 1: Perception (Vision)
# ==========================================

class PerceptionNode:
    def __init__(self):
        self.model_name = os.getenv("VLM_MODEL_NAME", "llava")
        raw_url = os.getenv("VLM_API_URL")
        self.api_url, self.api_url_fallback = build_endpoint(raw_url, "/perception", "/api/generate")
        self.max_frames = int(os.getenv("MAX_PERCEPTION_FRAMES", "12"))
        self.center_crop = os.getenv("PERCEPTION_CENTER_CROP", "1").strip().lower() in {"1", "true", "yes", "on"}
        self.resize_to = int(os.getenv("PERCEPTION_RESIZE_TO", "896"))

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
            keyframes = sampled
        
        if not keyframes:
            return "Error: No frames could be extracted from the video."

        system_prompt = (
            "You are an expert Foley artist and sound designer. "
            "I will provide you with a sequence of frames from a silent video. "
            "Describe the physical interactions, materials, and potential sounds in chronological order. "
            "Output the log with timestamps in this format: [0.0s] description of action."
        )

        images_base64 = []
        timestamp_context = "Timestamps for the provided frames:\n"
        
        for idx, kf in enumerate(keyframes):
            b64_img = self.encode_image_to_base64(kf["frame"])
            images_base64.append(b64_img)
            timestamp_context += f"Image {idx+1}: [{kf['timestamp']}s]\n"

        full_prompt = f"{system_prompt}\n\n{timestamp_context}"

        print(f"[Perception Node] Sending {len(images_base64)} frames to VLM ({self.model_name})...")

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
            return vlm_log
            
        except Exception as e:
            print(f"[Perception Node] API Error: {e}")
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

    def evaluate(self, prompt: str, audio_path: str) -> float:
        with open(audio_path, "rb") as fh:
            b64_audio = base64.b64encode(fh.read()).decode('utf-8')
            
        response = requests.post(self.api_url, json={"prompt": prompt, "audio_base64": b64_audio})
        if response.status_code == 404 and self.api_url_fallback:
            response = requests.post(self.api_url_fallback, json={"prompt": prompt, "audio_base64": b64_audio})
        if not response.ok:
            raise RuntimeError(f"Verification API error {response.status_code}: {response.text}")
        
        score = response.json().get("similarity_score", 0.0)
        print(f"[Verification Node] CLAP Score: {score:.2f} for '{prompt[:40]}...'")
        return score

# ==========================================
# Final Stage: Stitching & Orchestration
# ==========================================

class FoleyOrchestrator:
    def __init__(self):
        self.perception = PerceptionNode()
        self.planner = PlannerNode()
        self.execution = ExecutionNode()
        self.verification = VerificationNode(threshold=0.50)
        self.max_retries = 3
        self.max_video_seconds = float(os.getenv("MAX_VIDEO_SECONDS", "15"))
        self.video_output_dir = os.getenv("GENERATED_VIDEO_DIR", "./generated_outputs/video")
        self.temp_output_dir = os.getenv("GENERATED_TEMP_DIR", "./generated_outputs/tmp")
        os.makedirs(self.video_output_dir, exist_ok=True)
        os.makedirs(self.temp_output_dir, exist_ok=True)

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

    def run_pipeline(self, video_path: str, output_path: str):
        print(f"--- Starting Autonomous Foley Generation for {video_path} ---")
        if not os.path.isabs(output_path) and os.path.dirname(output_path) == "":
            output_path = os.path.join(self.video_output_dir, output_path)
        prepared_video_path, prepared_duration, is_temp = self.prepare_video(video_path)
        print(f"[Orchestrator] Using timeline duration: {prepared_duration:.2f}s")
        try:
            vlm_log = self.perception.analyze_video(prepared_video_path)
            if vlm_log.startswith("Error:"):
                raise RuntimeError(f"Perception failed: {vlm_log}")
            audio_plan = self.planner.create_audio_plan(vlm_log, prepared_duration)

            final_events = []
            for event in audio_plan:
                current_prompt = event.original_prompt
                success = False

                for attempt in range(1, self.max_retries + 1):
                    print(f"  -> Processing event at {event.timestamp_sec}s (Attempt {attempt}/{self.max_retries})")

                    audio_path = self.execution.generate_audio(current_prompt, event.timestamp_sec, event.duration_sec, attempt)
                    score = self.verification.evaluate(current_prompt, audio_path)

                    if score >= self.verification.threshold:
                        print("  -> [Critique Passed] Semantic match achieved.")
                        event.audio_file_path = audio_path
                        event.similarity_score = score
                        event.refined_prompt = current_prompt
                        success = True
                        break
                    else:
                        print("  -> [Critique Failed] Semantic mismatch. Retrying...")
                        current_prompt = self.planner.refine_prompt(current_prompt, score)

                if not success:
                    print(f"  -> [Warning] Event at {event.timestamp_sec}s failed to reach threshold after {self.max_retries} attempts. Using best effort.")
                    event.audio_file_path = audio_path

                final_events.append(event)
                print("-" * 40)

            self.stitch_audio_to_video(prepared_video_path, final_events, output_path)
        finally:
            if is_temp and os.path.exists(prepared_video_path):
                os.remove(prepared_video_path)

# ==========================================
# Execution Entry Point
# ==========================================
if __name__ == "__main__":
    orchestrator = FoleyOrchestrator()
    orchestrator.run_pipeline("input_video.mp4", "output_foley_video.mp4")
