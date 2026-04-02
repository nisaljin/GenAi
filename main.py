import cv2
import os
import json
import base64
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
        self.api_url = os.getenv("VLM_API_URL")

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

    def encode_image_to_base64(self, image: np.ndarray) -> str:
        _, buffer = cv2.imencode('.jpg', image)
        return base64.b64encode(buffer).decode('utf-8')

    def analyze_video(self, video_path: str) -> str:
        keyframes = self.extract_keyframes(video_path, threshold=15.0)
        
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

        payload = {
            "model": self.model_name,
            "prompt": full_prompt,
            "images": images_base64,
            "stream": False
        }

        try:
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            vlm_log = response.json().get("response", "")
            
            print(f"[Perception Node] Successfully generated VLM Log.")
            return vlm_log
            
        except requests.exceptions.RequestException as e:
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

    def create_audio_plan(self, vlm_log: str) -> List[AudioEvent]:
        prompt = (
            "You are a Foley Audio Director. Read this video log and output a JSON object containing a 'data' array of acoustic events. "
            "Strictly use this format: {\"data\": [{\"timestamp_sec\": 0.0, \"duration_sec\": 13.0, \"original_prompt\": \"Continuous ambient sound of...\"}]}. "
            "CRITICAL: If the scene has a continuous atmosphere (like a room or nature), create ONE long event that spans the whole duration. "
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
            
            # Map the new duration variable
            events = [AudioEvent(
                timestamp_sec=e["timestamp_sec"], 
                duration_sec=e.get("duration_sec", 2.0),
                original_prompt=e["original_prompt"], 
                refined_prompt=""
            ) for e in events_data]
            
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
        self.api_url = f"{base_url}/generate"
        self.output_dir = "./generated_audio"
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_audio(self, prompt: str, timestamp: float, duration: float, attempt: int) -> str:
        print(f"[Execution Node] Requesting {duration}s audio for: '{prompt[:40]}...'")
        
        # Send the duration variable to the RunPod server
        response = requests.post(self.api_url, json={"prompt": prompt, "duration": duration})
        response.raise_for_status()
        
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
        self.api_url = f"{base_url}/evaluate"

    def evaluate(self, prompt: str, audio_path: str) -> float:
        with open(audio_path, "rb") as fh:
            b64_audio = base64.b64encode(fh.read()).decode('utf-8')
            
        response = requests.post(self.api_url, json={"prompt": prompt, "audio_base64": b64_audio})
        response.raise_for_status()
        
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

    def stitch_audio_to_video(self, video_path: str, events: List[AudioEvent], output_path: str):
        print("[Stitcher] Assembling final video...")
        
        video_clip = VideoFileClip(video_path)
        base_audio = AudioSegment.silent(duration=int(video_clip.duration * 1000))

        for event in events:
            if event.audio_file_path and os.path.exists(event.audio_file_path):
                foley_clip = AudioSegment.from_wav(event.audio_file_path)
                insert_position_ms = int(event.timestamp_sec * 1000)
                base_audio = base_audio.overlay(foley_clip, position=insert_position_ms)

        temp_audio_path = "./temp_mixed_audio.wav"
        base_audio.export(temp_audio_path, format="wav")

        final_audio_clip = AudioFileClip(temp_audio_path)
        final_video = video_clip.with_audio(final_audio_clip)
        final_video.write_videofile(output_path, codec="libx264", audio_codec="aac")
        
        os.remove(temp_audio_path)
        print(f"[Stitcher] Success! Final video saved to {output_path}")

    def run_pipeline(self, video_path: str, output_path: str):
        print(f"--- Starting Autonomous Foley Generation for {video_path} ---")
        
        vlm_log = self.perception.analyze_video(video_path)
        audio_plan = self.planner.create_audio_plan(vlm_log)
        
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
            
        self.stitch_audio_to_video(video_path, final_events, output_path)

# ==========================================
# Execution Entry Point
# ==========================================
if __name__ == "__main__":
    orchestrator = FoleyOrchestrator()
    orchestrator.run_pipeline("input_video.mp4", "output_foley_video.mp4")