#!/usr/bin/env python3
"""
ASR Service - Speech Recognition Microservice
Uses OpenAI Whisper medium.en model for transcription
"""

import os
import logging
import tempfile
import re
from typing import Optional
from pathlib import Path
import subprocess

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import torch
import whisper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="ASR Service",
    description="Speech Recognition Service using OpenAI Whisper",
    version="2.0.0"
)

# Global model variable
model = None

class TranscriptionRequest(BaseModel):
    audio_data: Optional[str] = None
    language: Optional[str] = None

class TranscriptionResponse(BaseModel):
    text: str
    language: Optional[str] = None
    language_probability: Optional[float] = None
    duration: Optional[float] = None

def remove_repetition(text: str) -> str:
    """Remove repetitive phrases from transcribed text - generic version"""
    if not text or not isinstance(text, str):
        return text
    
    words = text.split()
    if len(words) < 3:
        return text
    
    result_words = []
    i = 0
    
    while i < len(words):
        # Try different phrase lengths starting from 1 word
        phrase_found = False
        
        for phrase_length in range(1, min(len(words) - i, 5) + 1):  # Max 5 words per phrase
            if i + phrase_length > len(words):
                break
                
            phrase = words[i:i + phrase_length]
            repeat_count = 1
            j = i + phrase_length
            
            # Count consecutive repetitions
            while j + phrase_length <= len(words):
                if words[j:j + phrase_length] == phrase:
                    repeat_count += 1
                    j += phrase_length
                else:
                    break
            
            # If we found 2 or more repetitions, keep only one instance
            if repeat_count >= 2:
                result_words.extend(phrase)
                i = j  # Skip all repetitions
                phrase_found = True
                break
        
        if not phrase_found:
            result_words.append(words[i])
            i += 1
    
    return " ".join(result_words)

def convert_to_wav_16k_mono(input_path: str, output_path: str) -> bool:
    """Convert audio to 16kHz mono WAV using ffmpeg"""
    try:
        cmd = [
            "ffmpeg", "-i", input_path,
            "-ar", "16000",  # 16kHz sample rate
            "-ac", "1",      # mono
            "-y",            # overwrite output
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"Audio converted successfully: {input_path} -> {output_path}")
            return True
        else:
            logger.error(f"FFmpeg conversion failed: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Audio conversion error: {e}")
        return False

def load_model():
    """Load the OpenAI Whisper model"""
    global model
    try:
        model_name = os.getenv("MODEL_NAME", "medium.en")
        device = os.getenv("DEVICE", "cpu")
        
        logger.info(f"Loading OpenAI Whisper model: {model_name} on {device}")
        
        # Load model with device specification
        if device == "cuda" and torch.cuda.is_available():
            model = whisper.load_model(model_name, device="cuda")
        else:
            model = whisper.load_model(model_name, device="cpu")
        
        logger.info("OpenAI Whisper model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup"""
    logger.info("Starting ASR Service...")
    if not load_model():
        logger.error("Failed to load model during startup")
        raise RuntimeError("Model loading failed")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_name": os.getenv("MODEL_NAME", "medium.en"),
        "device": os.getenv("DEVICE", "cpu")
    }

@app.get("/info")
async def get_info():
    """Get model information"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_name": os.getenv("MODEL_NAME", "medium.en"),
        "device": os.getenv("DEVICE", "cpu"),
        "model_loaded": True
    }

@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """Transcribe audio file using OpenAI Whisper"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        logger.info(f"Transcribing audio file: {file.filename}")
        
        # Read uploaded file
        audio_data = await file.read()
        logger.info(f"Uploaded content_type={file.content_type}, size_bytes={len(audio_data)}")
        
        # Create temporary file for input
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as temp_input:
            temp_input.write(audio_data)
            temp_input_path = temp_input.name
        
        # Create temporary file for converted audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_output:
            temp_output_path = temp_output.name
        
        try:
            # Convert to 16kHz mono WAV
            logger.info(f"Converting audio from {temp_input_path} to {temp_output_path}")
            if not convert_to_wav_16k_mono(temp_input_path, temp_output_path):
                logger.error("Audio conversion failed")
                raise HTTPException(status_code=400, detail="Audio conversion failed")
            logger.info(f"Audio conversion successful, output size: {os.path.getsize(temp_output_path)} bytes")
            
            # Transcribe using OpenAI Whisper with optimized parameters
            logger.info("Starting transcription with OpenAI Whisper...")
            logger.info(f"Audio file path: {temp_output_path}")
            logger.info(f"Audio file size: {os.path.getsize(temp_output_path)} bytes")
            
            result = model.transcribe(
                temp_output_path,
                beam_size=5,  # Beam search reduces repetition
                temperature=0.0,  # Deterministic output
                condition_on_previous_text=False,  # Don't condition on previous text
                fp16=False,  # Use fp32 for stability
                language="en",  # Force English language
                initial_prompt="Hello, hi, hey, good morning, good afternoon, good evening"  # Help with common greetings
            )
            
            # Extract transcription
            transcribed_text = result["text"].strip()
            logger.info(f"Raw transcription: {transcribed_text}")
            
            # Apply repetition removal
            transcribed_text = remove_repetition(transcribed_text)
            logger.info(f"Processed transcription: {transcribed_text}")
            
            # Extract language info if available
            language = result.get("language", "en")
            language_probability = None
            
            # Calculate duration if available
            duration = None
            if "segments" in result and result["segments"]:
                duration = result["segments"][-1]["end"]
            
            logger.info(f"Transcription completed successfully")
            
            return TranscriptionResponse(
                text=transcribed_text,
                language=language,
                language_probability=language_probability,
                duration=duration
            )
            
        finally:
            # Clean up temporary files
            background_tasks.add_task(os.unlink, temp_input_path)
            background_tasks.add_task(os.unlink, temp_output_path)
            
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

@app.post("/transcribe_text")
async def transcribe_text(request: TranscriptionRequest):
    """Transcribe base64 encoded audio data"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        import base64
        
        # Decode base64 audio data
        audio_bytes = base64.b64decode(request.audio_data)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(audio_bytes)
            temp_path = temp_file.name
        
        try:
            # Transcribe
            result = model.transcribe(
                temp_path,
                beam_size=5,
                temperature=0.0,
                condition_on_previous_text=False,
                fp16=False
            )
            
            transcribed_text = result["text"].strip()
            transcribed_text = remove_repetition(transcribed_text)
            
            return TranscriptionResponse(
                text=transcribed_text,
                language=result.get("language", "en")
            )
            
        finally:
            os.unlink(temp_path)
            
    except Exception as e:
        logger.error(f"Text transcription error: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        log_level="info"
    )