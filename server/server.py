from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import base64
import io
import scipy.io.wavfile
from diffusers import AudioLDM2Pipeline
from transformers import ClapModel, ClapProcessor
import librosa

app = FastAPI()

# 1. Load AudioLDM2
print("Loading AudioLDM2...")
audio_pipe = AudioLDM2Pipeline.from_pretrained("cvssp/audioldm2", torch_dtype=torch.float16)
audio_pipe.to("cuda")

# 2. Load CLAP
print("Loading CLAP...")
clap_model = ClapModel.from_pretrained("laion/larger_clap_general").to("cuda")
clap_processor = ClapProcessor.from_pretrained("laion/larger_clap_general")

class AudioRequest(BaseModel):
    prompt: str
    duration: float

class EvalRequest(BaseModel):
    prompt: str
    audio_base64: str

@app.post("/generate")
def generate_audio(req: AudioRequest):
    try:
        import numpy as np
        print(f"Generating {req.duration}s for: {req.prompt}")
        
        # Force Foley Mode & block garbage
        enhanced_prompt = f"{req.prompt}, high quality Foley sound effect, cinematic, crisp, detailed, 48kHz"
        neg_prompt = "music, speech, human voice, background noise, static, muffled, noisy, electronic, digital"

        # Generate audio with dynamic length
        audio = audio_pipe(
            prompt=enhanced_prompt,
            negative_prompt=neg_prompt,
            num_inference_steps=150, 
            guidance_scale=3.5,      
            audio_length_in_s=req.duration # Uses the LLM's requested length
        ).audios[0]
        
        # ==========================================
        # THE ANTI-SCREECH FIX
        # Normalize the raw AI floats into clean 16-bit PCM audio
        # ==========================================
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        audio_int16 = (audio * 32767).astype(np.int16)
        
        # Save and send back
        byte_io = io.BytesIO()
        scipy.io.wavfile.write(byte_io, rate=16000, data=audio_int16)
        b64_audio = base64.b64encode(byte_io.getvalue()).decode('utf-8')
        
        return {"audio_base64": b64_audio}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/evaluate")
def evaluate_audio(req: EvalRequest):
    try:
        # Decode base64 audio
        audio_bytes = base64.b64decode(req.audio_base64)
        audio_array, sr = librosa.load(io.BytesIO(audio_bytes), sr=48000)

        # Process through CLAP
        inputs = clap_processor(text=[req.prompt], audios=audio_array, return_tensors="pt", padding=True, sampling_rate=48000)
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = clap_model(**inputs)
            # Calculate cosine similarity
            logits_per_audio = outputs.logits_per_audio
            score = logits_per_audio.item() / 100.0 # Normalize slightly for evaluation
            
        return {"similarity_score": score}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))