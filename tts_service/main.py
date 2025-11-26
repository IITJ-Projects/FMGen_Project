#!/usr/bin/env python3
"""
TTS Service - Text-to-Speech Microservice
Uses MeloTTS for high-quality speech synthesis
"""

import os
import logging
import io
import tempfile
from typing import Optional, Generator
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import torch
import torchaudio
import numpy as np

# Audio optimization imports
try:
    from pydub import AudioSegment
    from pydub.utils import which
    PYDUB_AVAILABLE = True
    # Check if ffmpeg is available for Opus support
    OPUS_AVAILABLE = which("ffmpeg") is not None
except ImportError:
    PYDUB_AVAILABLE = False
    OPUS_AVAILABLE = False

try:
    import webrtcvad
    VAD_AVAILABLE = True
except ImportError:
    VAD_AVAILABLE = False

# MeloTTS imports (you may need to adjust based on actual MeloTTS API)
try:
    from melotts import MeloTTS
    MELOTTS_AVAILABLE = True
except ImportError:
    MELOTTS_AVAILABLE = False
    # Fallback to basic TTS for development
    from gtts import gTTS
    import pygame

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="TTS Service",
    description="Text-to-Speech Service using MeloTTS",
    version="1.0.0"
)

# Global model instance
tts_model = None

# Global defaults (env-configurable)
DEFAULT_TTS_SPEED = float(os.getenv("TTS_DEFAULT_SPEED", "1.3"))

# Audio optimization functions
def encode_opus_audio(audio_data: bytes, sample_rate: int = 22050, bitrate: int = 64) -> bytes:
    """Encode audio data using Opus codec for real-time streaming"""
    if not OPUS_AVAILABLE or not PYDUB_AVAILABLE:
        return audio_data  # Fallback to original data
    
    try:
        # Convert bytes to AudioSegment
        audio_segment = AudioSegment(
            data=audio_data,
            sample_width=2,  # 16-bit
            frame_rate=sample_rate,
            channels=1  # Mono
        )
        
        # Export as Opus
        opus_buffer = io.BytesIO()
        audio_segment.export(
            opus_buffer,
            format="opus",
            bitrate=f"{bitrate}k",
            parameters=["-ac", "1"]  # Mono
        )
        return opus_buffer.getvalue()
    except Exception as e:
        logger.warning(f"Opus encoding failed, using original audio: {e}")
        return audio_data

def chunk_audio_optimized(audio_data: bytes, chunk_duration_ms: int = 100, sample_rate: int = 22050) -> Generator[bytes, None, None]:
    """Generate optimized audio chunks for real-time streaming"""
    if not PYDUB_AVAILABLE:
        # Fallback: simple chunking
        chunk_size = (sample_rate * 2 * chunk_duration_ms) // 1000  # 2 bytes per sample
        for i in range(0, len(audio_data), chunk_size):
            yield audio_data[i:i + chunk_size]
        return
    
    try:
        # Convert to AudioSegment for precise chunking
        audio_segment = AudioSegment(
            data=audio_data,
            sample_width=2,
            frame_rate=sample_rate,
            channels=1
        )
        
        # Generate chunks
        chunk_length_ms = chunk_duration_ms
        for i in range(0, len(audio_segment), chunk_length_ms):
            chunk = audio_segment[i:i + chunk_length_ms]
            yield chunk.raw_data
    except Exception as e:
        logger.warning(f"Optimized chunking failed, using simple chunking: {e}")
        # Fallback to simple chunking
        chunk_size = (sample_rate * 2 * chunk_duration_ms) // 1000
        for i in range(0, len(audio_data), chunk_size):
            yield audio_data[i:i + chunk_size]

def detect_voice_activity(audio_data: bytes, sample_rate: int = 16000) -> bool:
    """Detect if audio contains voice activity using WebRTC VAD"""
    if not VAD_AVAILABLE:
        return True  # Assume voice activity if VAD not available
    
    try:
        vad = webrtcvad.Vad(2)  # Aggressiveness level 2 (0-3)
        
        # Convert to 16-bit PCM
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        
        # Process in 10ms frames (required by WebRTC VAD)
        frame_size = int(sample_rate * 0.01)  # 10ms frames
        frames = [audio_array[i:i + frame_size] for i in range(0, len(audio_array), frame_size)]
        
        # Check if any frame contains voice
        for frame in frames:
            if len(frame) == frame_size:
                if vad.is_speech(frame.tobytes(), sample_rate):
                    return True
        return False
    except Exception as e:
        logger.warning(f"VAD detection failed: {e}")
        return True  # Assume voice activity on error

def adapt_emotional_tone(text: str, emotional_tone: str, context: str = None) -> str:
    """Adapt text and TTS parameters based on emotional tone"""
    try:
        # Emotional tone mappings
        tone_adaptations = {
            "happy": {
                "speed": 1.1,
                "pitch_modifier": 1.05,
                "text_prefix": "",
                "text_suffix": " 😊"
            },
            "sad": {
                "speed": 0.9,
                "pitch_modifier": 0.95,
                "text_prefix": "",
                "text_suffix": " 😢"
            },
            "excited": {
                "speed": 1.2,
                "pitch_modifier": 1.1,
                "text_prefix": "",
                "text_suffix": " 🎉"
            },
            "calm": {
                "speed": 0.95,
                "pitch_modifier": 0.98,
                "text_prefix": "",
                "text_suffix": " 🧘"
            },
            "neutral": {
                "speed": 1.0,
                "pitch_modifier": 1.0,
                "text_prefix": "",
                "text_suffix": ""
            }
        }
        
        adaptation = tone_adaptations.get(emotional_tone, tone_adaptations["neutral"])
        
        # Apply text modifications
        adapted_text = f"{adaptation['text_prefix']}{text}{adaptation['text_suffix']}"
        
        # Log emotional adaptation
        logger.info(f"Emotional tone adaptation: {emotional_tone} -> speed: {adaptation['speed']}, pitch: {adaptation['pitch_modifier']}")
        
        return adapted_text, adaptation
        
    except Exception as e:
        logger.warning(f"Emotional tone adaptation failed: {e}")
        return text, {"speed": 1.0, "pitch_modifier": 1.0, "text_prefix": "", "text_suffix": ""}

def analyze_context_for_emotion(text: str, context: str = None) -> str:
    """Analyze text and context to determine appropriate emotional tone"""
    try:
        # Simple keyword-based emotion detection
        text_lower = text.lower()
        
        # Happy indicators
        happy_keywords = ["great", "awesome", "wonderful", "amazing", "fantastic", "excellent", "love", "happy", "joy"]
        if any(keyword in text_lower for keyword in happy_keywords):
            return "happy"
        
        # Sad indicators
        sad_keywords = ["sad", "sorry", "unfortunately", "problem", "issue", "difficult", "trouble", "help"]
        if any(keyword in text_lower for keyword in sad_keywords):
            return "sad"
        
        # Excited indicators
        excited_keywords = ["excited", "thrilled", "amazing", "incredible", "wow", "fantastic", "celebration"]
        if any(keyword in text_lower for keyword in excited_keywords):
            return "excited"
        
        # Calm indicators
        calm_keywords = ["relax", "calm", "peaceful", "gentle", "soft", "quiet", "meditation"]
        if any(keyword in text_lower for keyword in calm_keywords):
            return "calm"
        
        # Default to neutral
        return "neutral"
        
    except Exception as e:
        logger.warning(f"Context emotion analysis failed: {e}")
        return "neutral"

class TTSRequest(BaseModel):
    text: str
    voice_id: Optional[str] = "default"
    speed: Optional[float] = DEFAULT_TTS_SPEED
    sample_rate: Optional[int] = 22050  # Higher sample rate for better quality
    chunk_duration_ms: Optional[int] = 100  # Larger chunks for smoother audio
    use_opus: Optional[bool] = False  # Use WAV for better compatibility
    bitrate: Optional[int] = 64  # Higher bitrate for better quality
    emotional_tone: Optional[str] = "neutral"  # neutral, happy, sad, excited, calm
    context: Optional[str] = None  # Context for emotional adaptation

class TTSResponse(BaseModel):
    status: str
    message: str
    audio_length: Optional[float] = None

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    streaming: bool
    sample_rate: int

def load_model():
    """Load the TTS model"""
    global tts_model
    try:
        model_name = os.getenv("MODEL_NAME", "melo-tts")
        voice_id = os.getenv("VOICE_ID", "default")
        sample_rate = int(os.getenv("SAMPLE_RATE", "24000"))
        
        logger.info(f"Loading TTS model: {model_name} with voice {voice_id}")
        
        if MELOTTS_AVAILABLE:
            # Initialize MeloTTS
            tts_model = MeloTTS(
                model_name=model_name,
                voice_id=voice_id,
                sample_rate=sample_rate
            )
        else:
            # Fallback to gTTS for development
            logger.warning("MeloTTS not available, using gTTS fallback")
            tts_model = "gtts"
        
        logger.info("TTS model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to load TTS model: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup"""
    logger.info("Starting TTS Service...")
    success = load_model()
    if not success:
        logger.error("Failed to load TTS model during startup")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if tts_model is not None else "unhealthy",
        model_loaded=tts_model is not None,
        streaming=os.getenv("STREAMING", "true").lower() == "true",
        sample_rate=int(os.getenv("SAMPLE_RATE", "24000"))
    )

@app.post("/speak")
async def speak_text(request: TTSRequest):
    """
    Convert text to speech and return audio stream
    """
    if tts_model is None:
        raise HTTPException(status_code=503, detail="TTS model not loaded")
    
    try:
        logger.info(f"Generating speech for text: {request.text[:50]}...")
        
        if MELOTTS_AVAILABLE and tts_model != "gtts":
            # Use MeloTTS
            audio_data = tts_model.synthesize(
                text=request.text,
                voice_id=request.voice_id,
                speed=request.speed
            )
            
            # Convert to WAV format
            audio_tensor = torch.tensor(audio_data).unsqueeze(0)
            
            # Save to temporary buffer
            buffer = io.BytesIO()
            torchaudio.save(buffer, audio_tensor, request.sample_rate, format="wav")
            buffer.seek(0)
            
            logger.info(f"Speech generated successfully: {len(audio_data)} samples")
            
            # Return raw bytes with proper headers
            from fastapi.responses import Response
            return Response(
                content=buffer.getvalue(),
                media_type="audio/wav",
                headers={
                    "Content-Disposition": f"attachment; filename=speech.wav"
                }
            )
            
        else:
            # Fallback to gTTS
            logger.info("Using gTTS fallback")
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
                tts = gTTS(text=request.text, lang='en', slow=False)
                tts.save(temp_file.name)
                
                # Read the file
                with open(temp_file.name, 'rb') as f:
                    audio_data = f.read()
                
                # Clean up
                os.unlink(temp_file.name)
            
            # Return raw bytes with proper headers
            from fastapi.responses import Response
            return Response(
                content=audio_data,
                media_type="audio/mpeg",
                headers={
                    "Content-Disposition": f"attachment; filename=speech.mp3"
                }
            )
            
    except Exception as e:
        logger.error(f"TTS generation error: {e}")
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")

@app.post("/speak_stream")
async def speak_text_stream(request: TTSRequest):
    """
    Stream audio chunks for real-time playback
    """
    if tts_model is None:
        raise HTTPException(status_code=503, detail="TTS model not loaded")
    
    try:
        logger.info(f"Streaming speech for text: {request.text[:50]}...")
        
        def generate_audio_chunks():
            """Generate optimized audio chunks for real-time streaming"""
            try:
                # Analyze emotional tone if not provided
                if request.emotional_tone == "neutral" and request.context:
                    detected_tone = analyze_context_for_emotion(request.text, request.context)
                    if detected_tone != "neutral":
                        request.emotional_tone = detected_tone
                
                # Adapt text and parameters based on emotional tone
                adapted_text, tone_adaptation = adapt_emotional_tone(
                    request.text, 
                    request.emotional_tone, 
                    request.context
                )
                
                # Apply emotional speed adaptation
                adapted_speed = request.speed * tone_adaptation["speed"]
                
                if MELOTTS_AVAILABLE and tts_model != "gtts":
                    # Use MeloTTS for streaming with emotional adaptation
                    audio_data = tts_model.synthesize(
                        text=adapted_text,
                        voice_id=request.voice_id,
                        speed=adapted_speed
                    )
                    
                    # Convert to WAV format
                    audio_tensor = torch.tensor(audio_data).unsqueeze(0)
                    buffer = io.BytesIO()
                    torchaudio.save(buffer, audio_tensor, request.sample_rate, format="wav")
                    buffer.seek(0)
                    raw_audio = buffer.getvalue()
                    
                    # Apply voice activity detection
                    if not detect_voice_activity(raw_audio, request.sample_rate):
                        logger.info("No voice activity detected, skipping audio generation")
                        return
                    
                    # Encode with Opus if requested
                    if request.use_opus:
                        raw_audio = encode_opus_audio(raw_audio, request.sample_rate, request.bitrate)
                    
                    # Generate optimized chunks
                    for chunk in chunk_audio_optimized(
                        raw_audio, 
                        request.chunk_duration_ms, 
                        request.sample_rate
                    ):
                        yield chunk
                        
                else:
                    # For fallback, generate full audio and chunk it
                    tts = gTTS(text=request.text, lang='en', slow=False)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
                        tts.save(temp_file.name)
                        with open(temp_file.name, 'rb') as f:
                            audio_data = f.read()
                        
                        # Apply voice activity detection
                        if not detect_voice_activity(audio_data, request.sample_rate):
                            logger.info("No voice activity detected, skipping audio generation")
                            return
                        
                        # Encode with Opus if requested
                        if request.use_opus:
                            audio_data = encode_opus_audio(audio_data, request.sample_rate, request.bitrate)
                        
                        # Generate optimized chunks
                        for chunk in chunk_audio_optimized(
                            audio_data, 
                            request.chunk_duration_ms, 
                            request.sample_rate
                        ):
                            yield chunk
                        
                        os.unlink(temp_file.name)
                        
            except Exception as e:
                logger.error(f"Audio chunking error: {e}")
                # Return empty chunk to prevent hanging
                yield b""
        
        # Determine media type based on codec
        media_type = "audio/opus" if request.use_opus else "audio/wav"
        content_type = "audio/opus" if request.use_opus else "audio/wav"
        
        return StreamingResponse(
            generate_audio_chunks(),
            media_type=media_type,
            headers={
                "Content-Type": content_type,
                "Transfer-Encoding": "chunked",
                "Cache-Control": "no-cache",
                "X-Audio-Sample-Rate": str(request.sample_rate),
                "X-Audio-Chunk-Duration": str(request.chunk_duration_ms),
                "X-Audio-Codec": "opus" if request.use_opus else "wav"
            }
        )
        
    except Exception as e:
        logger.error(f"TTS streaming error: {e}")
        raise HTTPException(status_code=500, detail=f"TTS streaming failed: {str(e)}")

@app.post("/speak_status", response_model=TTSResponse)
async def speak_status(request: TTSRequest):
    """
    Check if text can be synthesized (without generating audio)
    """
    if tts_model is None:
        raise HTTPException(status_code=503, detail="TTS model not loaded")
    
    try:
        # Simple validation
        if not request.text.strip():
            raise ValueError("Empty text provided")
        
        if len(request.text) > 1000:
            raise ValueError("Text too long (max 1000 characters)")
        
        return TTSResponse(
            status="ready",
            message="Text can be synthesized",
            audio_length=len(request.text) * 0.06  # Rough estimate
        )
        
    except Exception as e:
        logger.error(f"TTS validation error: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid request: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8003,
        reload=False,
        log_level="info"
    )
