#!/usr/bin/env python3
"""
Orchestration Service - Conversational AI Pipeline Coordinator
Manages ASR → RAG → LLM → TTS pipeline using Vocode
"""

import os
import logging
import asyncio
import json
import uuid
import time
from typing import Optional, Dict, Any, List
import base64
from datetime import datetime

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import StreamingResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel
import httpx
import aiofiles

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# WebRTC support (optional)
try:
    import aiortc
    from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate
    WEBRTC_AVAILABLE = True
    logger.info("WebRTC support enabled")
except ImportError:
    WEBRTC_AVAILABLE = False
    logger.info("WebRTC support disabled - using WebSocket fallback")

# Default system context for Zevo AI (optimized for speed)
DEFAULT_SYSTEM_CONTEXT = (
    "You are Zevo AI, a helpful voice assistant. Be conversational, friendly, and professional. "
    "Default to short, concise answers (1-2 sentences). If the user explicitly asks for more detail, "
    "depth, steps, or a comprehensive explanation, then provide a detailed answer. If you don't know "
    "something, say so and suggest alternatives. Format responses in Markdown when helpful, using "
    "headings, bullet lists, tables, and fenced code blocks."
)

# Heuristic to determine response length based on user's message intent
def determine_response_length(user_text: str) -> Dict[str, Any]:
    try:
        text = (user_text or "").lower()
        detailed_triggers = [
            "in detail", "detailed", "explain", "why", "how", "steps", "step-by-step",
            "comprehensive", "elaborate", "deep", "thorough", "long answer", "expand"
        ]
        brief_triggers = ["brief", "short", "concise", "tl;dr", "summary", "summarize"]

        wants_detailed = any(k in text for k in detailed_triggers)
        wants_brief = any(k in text for k in brief_triggers)

        # Longer questions may deserve more room
        long_question = len(text) > 220 or text.count("?") > 1

        if wants_brief:
            return {"max_tokens": 96, "temperature": 0.7, "top_p": 0.95, "top_k": 40}
        if wants_detailed or long_question:
            return {"max_tokens": 2048, "temperature": 0.7, "top_p": 0.95, "top_k": 40}
        # Default concise
        return {"max_tokens": 240, "temperature": 0.7, "top_p": 0.95, "top_k": 40}
    except Exception:
        # Safe fallback
        return {"max_tokens": 240, "temperature": 0.7, "top_p": 0.95, "top_k": 40}

# In-memory conversation storage (in production, use Redis or database)
conversation_memory = {}

# Performance monitoring storage
performance_history = []

# WebSocket connection management
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket
        logger.info(f"WebSocket connected for session: {session_id}")
    
    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
            logger.info(f"WebSocket disconnected for session: {session_id}")
    
    async def send_message(self, session_id: str, message: dict):
        if session_id in self.active_connections:
            try:
                await self.active_connections[session_id].send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error sending WebSocket message: {e}")
                self.disconnect(session_id)

manager = ConnectionManager()

# WebRTC connection management
if WEBRTC_AVAILABLE:
    webrtc_connections: Dict[str, RTCPeerConnection] = {}
else:
    webrtc_connections: Dict[str, Any] = {}

class LatencyTracker:
    """Track latency for each step in the conversation pipeline"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.start_time = time.time()
        self.steps = {}
        self.current_step = None
        self.step_start_time = None
    
    def start_step(self, step_name: str):
        """Start timing a pipeline step"""
        if self.current_step:
            self.end_step()
        
        self.current_step = step_name
        self.step_start_time = time.time()
        logger.info(f"⏱️  Starting {step_name} for session {self.session_id}")
    
    def end_step(self):
        """End timing the current step"""
        if self.current_step and self.step_start_time:
            duration = time.time() - self.step_start_time
            self.steps[self.current_step] = {
                "duration_ms": round(duration * 1000, 2),
                "start_time": self.step_start_time,
                "end_time": time.time()
            }
            logger.info(f"⏱️  {self.current_step} completed in {duration*1000:.2f}ms")
            self.current_step = None
            self.step_start_time = None
    
    def get_total_duration(self) -> float:
        """Get total pipeline duration in milliseconds"""
        return round((time.time() - self.start_time) * 1000, 2)
    
    def get_report(self) -> Dict[str, Any]:
        """Get comprehensive latency report"""
        self.end_step()  # End any current step
        
        total_duration = self.get_total_duration()
        
        # Calculate percentages
        step_percentages = {}
        for step_name, step_data in self.steps.items():
            percentage = (step_data["duration_ms"] / total_duration) * 100 if total_duration > 0 else 0
            step_percentages[step_name] = round(percentage, 1)
        
        return {
            "session_id": self.session_id,
            "total_duration_ms": total_duration,
            "steps": self.steps,
            "step_percentages": step_percentages,
            "timestamp": datetime.now().isoformat(),
            "bottleneck": max(self.steps.items(), key=lambda x: x[1]["duration_ms"])[0] if self.steps else None
        }

# Vocode availability check (lightweight)
try:
    import vocode  # noqa: F401
    VOCODE_AVAILABLE = True
    logger.info("Vocode package detected")
except Exception as e:
    VOCODE_AVAILABLE = False
    logger.warning(f"Vocode not available: {e}. Proceeding without Vocode.")

# Middleware to suppress uvicorn access logs for health checks
class SuppressHealthLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Suppress uvicorn access logging for health endpoints
        if request.url.path == "/health":
            import logging
            uvicorn_access_logger = logging.getLogger("uvicorn.access")
            original_level = uvicorn_access_logger.level
            uvicorn_access_logger.setLevel(logging.ERROR)
            try:
                response = await call_next(request)
                return response
            finally:
                uvicorn_access_logger.setLevel(original_level)
        else:
            return await call_next(request)

# Initialize FastAPI app
app = FastAPI(
    title="Orchestration Service",
    description="Conversational AI Pipeline Orchestrator",
    version="1.0.0"
)

# Add middleware to suppress health check logging
app.add_middleware(SuppressHealthLogMiddleware)

def safe_header_value(value: Optional[str]) -> str:
    """Sanitize header values to ASCII-safe, single-line strings."""
    if value is None:
        return ""
    try:
        s = str(value)
    except Exception:
        s = ""
    # Remove CR/LF and non-printable chars
    s = s.replace("\r", "").replace("\n", "")
    return s

# CORS for browser frontend
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "*")
# If wildcard, browsers disallow credentials; only enable credentials for explicit origins
allow_origins = [FRONTEND_ORIGIN] if FRONTEND_ORIGIN != "*" else ["*"]
allow_credentials = FRONTEND_ORIGIN != "*"
# Temporarily commented out to debug header issues
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=allow_origins,
#     allow_credentials=allow_credentials,
#     allow_methods=["*"],
#     allow_headers=["*"]
# )

# Service start time for status endpoint
SERVICE_START_TIME = datetime.utcnow()

# Service URLs from environment
ASR_SERVICE_URL = os.getenv("ASR_SERVICE_URL", "http://asr-service:8001")
LLM_SERVICE_URL = os.getenv("LLM_SERVICE_URL", "http://llm-service:8002")
RAG_SERVICE_URL = os.getenv("RAG_SERVICE_URL", "http://rag-service:8004")
TTS_SERVICE_URL = os.getenv("TTS_SERVICE_URL", "http://tts-service:8003")
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant-db")
QDRANT_PORT = os.getenv("QDRANT_PORT", "6333")

# Global HTTP client
http_client = None

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    language: Optional[str] = "en"
    context: Optional[str] = None

class TextChatRequest(BaseModel):
    message: str
    session_id: str
    conversation_history: Optional[List[Dict[str, Any]]] = []
    read_aloud: Optional[bool] = False

class ChatResponse(BaseModel):
    session_id: str
    response_text: str
    audio_url: Optional[str] = None
    timestamp: str
    latency_report: Optional[Dict[str, Any]] = None

class ConversationStartRequest(BaseModel):
    user_id: str
    language: str = "en"
    context: Optional[str] = None

class ConversationStartResponse(BaseModel):
    session_id: str
    status: str
    message: str

# WebRTC Models
class WebRTCOfferRequest(BaseModel):
    session_id: str
    offer: Dict[str, Any]

class WebRTCOfferResponse(BaseModel):
    answer: Dict[str, Any]
    status: str

class WebRTCIceCandidateRequest(BaseModel):
    session_id: str
    candidate: Dict[str, Any]

class HealthResponse(BaseModel):
    status: str
    services: Dict[str, str]
    vocode_available: bool

class ServiceStatus(BaseModel):
    asr: str
    llm: str
    rag: str
    tts: str
    qdrant: str

class SpeakRequest(BaseModel):
    text: str
    voice_id: Optional[str] = "default"
    speed: Optional[float] = 1.0
    sample_rate: Optional[int] = 22050
    emotional_tone: Optional[str] = "neutral"

async def get_http_client():
    """Get or create HTTP client"""
    global http_client
    if http_client is None:
        http_client = httpx.AsyncClient(timeout=30.0)
    return http_client

# Health check cache to reduce frequency of calls
_health_check_cache = {}
_health_check_cache_ttl = 60  # Cache for 60 seconds

async def check_service_health(service_url: str, service_name: str) -> str:
    """Check if a service is healthy with caching to reduce frequency"""
    cache_key = service_name.lower()
    current_time = time.time()
    
    # Return cached result if still valid
    if cache_key in _health_check_cache:
        cached_time, cached_status = _health_check_cache[cache_key]
        if current_time - cached_time < _health_check_cache_ttl:
            return cached_status
    
    try:
        client = await get_http_client()
        # Suppress httpx INFO logging for health checks
        import logging as httpx_logging
        original_level = httpx_logging.getLogger("httpx").level
        httpx_logging.getLogger("httpx").setLevel(logging.WARNING)
        
        try:
            response = await client.get(f"{service_url}/health", timeout=5.0)
            if response.status_code == 200:
                status = "healthy"
            else:
                status = f"unhealthy (status: {response.status_code})"
        finally:
            httpx_logging.getLogger("httpx").setLevel(original_level)
        
        # Cache the result
        _health_check_cache[cache_key] = (current_time, status)
        return status
    except Exception as e:
        status = f"unhealthy (error: {str(e)})"
        # Only log errors, not every check
        logger.debug(f"Health check failed for {service_name}: {e}")
        _health_check_cache[cache_key] = (current_time, status)
        return status

async def check_qdrant_health(base_url: str) -> str:
    """Check Qdrant health with fallback endpoints and caching"""
    cache_key = "qdrant"
    current_time = time.time()
    
    # Return cached result if still valid
    if cache_key in _health_check_cache:
        cached_time, cached_status = _health_check_cache[cache_key]
        if current_time - cached_time < _health_check_cache_ttl:
            return cached_status
    
    try:
        client = await get_http_client()
        # Suppress httpx INFO logging for health checks
        import logging as httpx_logging
        original_level = httpx_logging.getLogger("httpx").level
        httpx_logging.getLogger("httpx").setLevel(logging.WARNING)
        
        try:
            # Try health endpoints
            for path in ["/healthz", "/readyz"]:
                try:
                    resp = await client.get(f"{base_url}{path}", timeout=5.0)
                    if resp.status_code == 200:
                        status = "healthy"
                        _health_check_cache[cache_key] = (current_time, status)
                        httpx_logging.getLogger("httpx").setLevel(original_level)
                        return status
                except Exception:
                    pass
            # Fallback to collections listing
            resp = await client.get(f"{base_url}/collections", timeout=5.0)
            if resp.status_code == 200:
                status = "healthy"
            else:
                status = f"unhealthy (status: {resp.status_code})"
        finally:
            httpx_logging.getLogger("httpx").setLevel(original_level)
        
        # Cache the result
        _health_check_cache[cache_key] = (current_time, status)
        return status
    except Exception as e:
        status = f"unhealthy (error: {str(e)})"
        # Only log errors at debug level
        logger.debug(f"Qdrant health check failed: {e}")
        _health_check_cache[cache_key] = (current_time, status)
        return status

@app.on_event("startup")
async def startup_event():
    """Initialize the orchestration service"""
    logger.info("Starting Orchestration Service...")
    logger.info(f"ASR Service: {ASR_SERVICE_URL}")
    logger.info(f"LLM Service: {LLM_SERVICE_URL}")
    logger.info(f"RAG Service: {RAG_SERVICE_URL}")
    logger.info(f"TTS Service: {TTS_SERVICE_URL}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global http_client
    if http_client:
        await http_client.aclose()

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint - cached to reduce logging frequency"""
    services = {}
    
    # Check all services (with caching)
    services["asr"] = await check_service_health(ASR_SERVICE_URL, "ASR")
    services["llm"] = await check_service_health(LLM_SERVICE_URL, "LLM")
    services["rag"] = await check_service_health(RAG_SERVICE_URL, "RAG")
    services["tts"] = await check_service_health(TTS_SERVICE_URL, "TTS")
    services["qdrant"] = await check_qdrant_health(f"http://{QDRANT_HOST}:{QDRANT_PORT}")
    
    # Overall status
    all_healthy = all(status == "healthy" for status in services.values())
    
    # Only log if unhealthy to reduce log noise
    if not all_healthy:
        logger.warning(f"Health check shows degraded services: {services}")
    
    return HealthResponse(
        status="healthy" if all_healthy else "degraded",
        services=services,
        vocode_available=VOCODE_AVAILABLE
    )

@app.post("/conversation/start", response_model=ConversationStartResponse)
async def start_conversation(request: ConversationStartRequest):
    """Start a new conversation session"""
    try:
        session_id = str(uuid.uuid4())
        
        # Store session info (in production, use Redis or database)
        session_data = {
            "session_id": session_id,
            "user_id": request.user_id,
            "language": request.language,
            "context": request.context,
            "created_at": datetime.utcnow().isoformat(),
            "messages": []
        }
        
        # Save session (implement proper storage)
        await save_session(session_id, session_data)
        
        logger.info(f"Started conversation session: {session_id} for user: {request.user_id}")
        
        return ConversationStartResponse(
            session_id=session_id,
            status="started",
            message="Conversation session created successfully"
        )
        
    except Exception as e:
        logger.error(f"Failed to start conversation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start conversation: {str(e)}")

@app.post("/chat")
async def chat_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    session_id: Optional[str] = None,
    user_id: Optional[str] = None
):
    """
    Main chat endpoint - uses Vocode pipeline for orchestration
    Pipeline: ASR → RAG → LLM → TTS (via Vocode)
    """
    if not session_id:
        session_id = str(uuid.uuid4())
    
    try:
        logger.info(f"Processing chat request for session: {session_id}")
        logger.info("=== RUNNING NEW VOCODE PIPELINE VERSION 2.0 ===")

        if not VOCODE_AVAILABLE:
            logger.error("Vocode not available - cannot process request")
            raise HTTPException(status_code=503, detail="Vocode not available")

        # Use Vocode pipeline for orchestration
        logger.info("Using Vocode pipeline for orchestration")
        pipeline = await create_vocode_pipeline()
        response = await process_with_vocode(pipeline, file, session_id)
        
        # Convert Vocode response to frontend-compatible format
        if hasattr(response, 'audio_data') and response.audio_data:
            # Use audio data directly from Vocode pipeline
            audio_bytes = response.audio_data
        else:
            # Generate audio from text response
            audio_bytes = await text_to_speech(response.response_text)
        
        response_content = {
            "session_id": str(session_id),
            "audio_base64": base64.b64encode(audio_bytes).decode("utf-8")
        }
        
        return Response(
            content=json.dumps(response_content),
            media_type="application/json"
        )

    except HTTPException:
        # Re-raise HTTP exceptions (already logged)
        raise
    except Exception as e:
        logger.error(f"Chat processing error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Chat processing failed: {str(e)}"
        )

@app.post("/api/chat")
async def text_chat_endpoint(request: TextChatRequest):
    """
    Text-based chat endpoint with conversation history
    Pipeline: RAG → LLM → TTS (text input, audio output)
    """
    # Initialize latency tracker
    latency_tracker = LatencyTracker(request.session_id)
    
    try:
        logger.info(f"Processing text chat for session: {request.session_id}")
        logger.info(f"Message: {request.message}")
        logger.info(f"Conversation history length: {len(request.conversation_history)}")

        # Step 1: Store conversation history
        latency_tracker.start_step("conversation_storage")
        if request.session_id not in conversation_memory:
            conversation_memory[request.session_id] = []
        
        # Add current message to history
        conversation_memory[request.session_id].append({
            "role": "user",
            "content": request.message,
            "timestamp": datetime.now().isoformat()
        })
        latency_tracker.end_step()

        # Step 2: Retrieve relevant context from RAG
        latency_tracker.start_step("rag_retrieval")
        rag_context = ""
        try:
            logger.info(f"🔍 RAG: Retrieving context for query: '{request.message}'")
            async with httpx.AsyncClient(timeout=10.0) as client:
                rag_request = {
                    "query": request.message,
                    "top_k": 5,
                    "use_reranker": True,
                    "language": "en"
                }
                
                rag_response = await client.post(
                    f"{RAG_SERVICE_URL}/retrieve",
                    json=rag_request
                )
                
                if rag_response.status_code == 200:
                    rag_result = rag_response.json()
                    if rag_result.get("documents") and len(rag_result["documents"]) > 0:
                        # Combine retrieved document texts into context
                        rag_context = "\n\n".join([
                            f"Relevant information: {doc.get('text', '')}" 
                            for doc in rag_result["documents"][:3]  # Top 3 documents
                        ])
                        logger.info(f"🔍 RAG: ✅ Retrieved {len(rag_result['documents'])} relevant documents")
                        logger.info(f"🔍 RAG: 📄 Document count in response: {rag_result.get('total_found', 0)}")
                        logger.info(f"🔍 RAG: ⏱️ Processing time: {rag_result.get('processing_time', 0):.3f}s")
                        # Log first document preview
                        if rag_result["documents"]:
                            first_doc = rag_result["documents"][0]
                            preview = first_doc.get('text', '')[:100] + "..." if len(first_doc.get('text', '')) > 100 else first_doc.get('text', '')
                            logger.info(f"🔍 RAG: 📝 Top document preview: {preview}")
                    else:
                        logger.info(f"🔍 RAG: ⚠️ No relevant documents found for query '{request.message}'")
                        logger.info(f"🔍 RAG: 📊 Response: {rag_result}")
                else:
                    logger.warning(f"🔍 RAG: ⚠️ RAG service returned status {rag_response.status_code}")
                    logger.warning(f"🔍 RAG: Response text: {rag_response.text[:200]}")
        except Exception as e:
            logger.warning(f"🔍 RAG: ❌ Error retrieving context: {e} - continuing without RAG")
        latency_tracker.end_step()

        # Step 3: Create context from RAG + conversation history
        # PRIORITY: RAG context comes FIRST and is emphasized
        latency_tracker.start_step("context_preparation")
        context_parts = []
        
        # CRITICAL: RAG context gets highest priority and explicit instructions
        if rag_context:
            # Strong instruction to prioritize RAG over general knowledge
            context_parts.append("=== CRITICAL: PRIORITIZE KNOWLEDGE BASE INFORMATION ===")
            context_parts.append("Use the following information from the knowledge base as the PRIMARY source.")
            context_parts.append("Only use your general knowledge if the knowledge base doesn't contain relevant information.")
            context_parts.append("When knowledge base information is available, it takes precedence over general knowledge.")
            context_parts.append("\n--- KNOWLEDGE BASE INFORMATION (HIGHEST PRIORITY) ---")
            context_parts.append(rag_context)
            context_parts.append("--- END KNOWLEDGE BASE INFORMATION ---\n")
            logger.info(f"📚 Context: Added RAG context with HIGH PRIORITY ({len(rag_context)} chars) to LLM prompt")
        else:
            logger.info(f"📚 Context: No RAG context available - using only conversation history")
        
        # Add system context after RAG (lower priority)
        context_parts.append("\n--- SYSTEM CONTEXT ---")
        context_parts.append(DEFAULT_SYSTEM_CONTEXT)
        
        # Add recent conversation context (last 2 exchanges for speed) - lowest priority
        recent_history = conversation_memory[request.session_id][-4:]  # Last 4 messages (2 exchanges)
        if recent_history:
            context_parts.append("\n--- RECENT CONVERSATION (FOR CONTEXT ONLY) ---")
            for msg in recent_history:
                role = "User" if msg["role"] == "user" else "Assistant"
                # Truncate long messages for faster processing
                content = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
                context_parts.append(f"{role}: {content}")
            context_parts.append("--- END CONVERSATION CONTEXT ---")
        
        context = "\n".join(context_parts)
        latency_tracker.end_step()

        # Step 4: Generate response using LLM
        latency_tracker.start_step("llm_generation")
        logger.info("🤖 STEP 1 - LLM: Starting response generation")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            gen_params = determine_response_length(request.message)
            llm_request = {
                "prompt": request.message,
                "context": context,
                **gen_params
            }
            
            llm_response = await client.post(
                f"http://llm-service:8002/generate",
                json=llm_request
            )
            
            if llm_response.status_code != 200:
                raise HTTPException(status_code=500, detail="LLM service error")
            
            llm_result = llm_response.json()
            response_text = llm_result["response"]

            # Ensure the response ends with terminal punctuation
            if response_text and not response_text.rstrip().endswith((".", "!", "?")):
                response_text = response_text.rstrip() + "."
            
            # Ensure response is complete (not truncated)
            if response_text and not response_text.rstrip().endswith(('.', '!', '?')):
                # If response doesn't end with proper punctuation, it might be truncated
                logger.warning(f"🤖 STEP 1 - LLM: Response may be truncated: {response_text[-50:]}")
            
            logger.info(f"🤖 STEP 1 - LLM: ✅ SUCCESS - Generated response")
        latency_tracker.end_step()

        # Step 5: Store AI response in history
        latency_tracker.start_step("response_storage")
        conversation_memory[request.session_id].append({
            "role": "assistant",
            "content": response_text,
            "timestamp": datetime.now().isoformat()
        })
        latency_tracker.end_step()

        # Step 6: Optionally generate audio using TTS (simple, non-streaming)
        audio_base64 = None
        if request.read_aloud:
            latency_tracker.start_step("tts_generation")
            logger.info("🔊 STEP 2 - TTS: Starting audio synthesis (read_aloud=true)")
            async with httpx.AsyncClient(timeout=30.0) as client:
                tts_request = {
                    "text": response_text,
                    "voice_id": "default",
                    "sample_rate": 22050,
                    "chunk_duration_ms": 100,
                    "use_opus": False,
                    "bitrate": 64,
                    "emotional_tone": "neutral",
                    "context": f"Conversation response for session {request.session_id}"
                }
                tts_response = await client.post(
                    f"http://tts-service:8003/speak",
                    json=tts_request
                )
                if tts_response.status_code == 200:
                    audio_bytes = tts_response.content
                    audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
                    logger.info("🔊 STEP 2 - TTS: ✅ SUCCESS - Generated audio")
                else:
                    logger.warning("TTS service error - returning text only")
                    audio_base64 = None
            latency_tracker.end_step()

        # Step 6: Prepare response
        latency_tracker.start_step("response_preparation")
        latency_report = latency_tracker.get_report()
        
        # Store performance data for monitoring
        performance_history.append(latency_report)
        if len(performance_history) > 100:  # Keep last 100 requests
            performance_history.pop(0)
        
        # Return response
        response_data = {
            "response": response_text,
            "transcribed_text": request.message,  # Echo back the input
            "audio_base64": audio_base64,
            "session_id": request.session_id,
            "latency_report": latency_report
        }
        latency_tracker.end_step()

        return Response(
            content=json.dumps(response_data),
            media_type="application/json"
        )
        
    except Exception as e:
        logger.error(f"Text chat processing error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Text chat processing failed: {str(e)}"
        )

@app.post("/api/tts")
async def synthesize_text(request: SpeakRequest):
    """
    Simple TTS endpoint: accepts text and returns base64 audio.
    Keeps Text Mode simple: on-demand playback via a speaker icon.
    """
    try:
        if not request.text or not request.text.strip():
            raise HTTPException(status_code=400, detail="Text is required")

        logger.info(f"🔊 /api/tts: Received request - text length: {len(request.text)}, text preview: {request.text[:100]}")
        logger.info(f"🔊 /api/tts: Calling TTS service at {TTS_SERVICE_URL}/speak")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            tts_request = {
                "text": request.text,
                "voice_id": request.voice_id or "default",
                "speed": request.speed or 1.0,
                "sample_rate": request.sample_rate or 22050,
                "chunk_duration_ms": 100,
                "use_opus": False,
                "bitrate": 64,
                "emotional_tone": request.emotional_tone or "neutral",
                "context": "On-demand read aloud for text mode"
            }

            try:
                tts_response = await client.post(
                    f"{TTS_SERVICE_URL}/speak",
                    json=tts_request
                )
                
                logger.info(f"🔊 TTS response status: {tts_response.status_code}, content-type: {tts_response.headers.get('content-type', 'unknown')}, content-length: {len(tts_response.content) if tts_response.content else 0}")
                
                if tts_response.status_code != 200:
                    error_detail = f"TTS service returned status {tts_response.status_code}"
                    try:
                        error_body = tts_response.text[:500]  # First 500 chars
                        error_detail += f": {error_body}"
                    except:
                        pass
                    logger.error(f"❌ TTS service error: {error_detail}")
                    raise HTTPException(status_code=502, detail=error_detail)

                audio_bytes = tts_response.content
                if not audio_bytes:
                    logger.error("❌ TTS service returned empty response")
                    raise HTTPException(status_code=502, detail="TTS service returned empty audio")
                
                audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
                logger.info(f"✅ TTS success: Generated {len(audio_base64)} chars of base64 audio")
                return {"audio_base64": audio_base64}
                
            except httpx.TimeoutException:
                logger.error(f"❌ TTS service timeout: {TTS_SERVICE_URL}/speak")
                raise HTTPException(status_code=504, detail="TTS service timeout")
            except httpx.ConnectError as e:
                logger.error(f"❌ TTS service connection error: {e}")
                raise HTTPException(status_code=503, detail=f"TTS service unreachable: {str(e)}")
            except httpx.HTTPError as e:
                logger.error(f"❌ TTS service HTTP error: {e}")
                raise HTTPException(status_code=502, detail=f"TTS service HTTP error: {str(e)}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ /api/tts failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to synthesize speech: {str(e)}")

@app.get("/api/latency/{session_id}")
async def get_latency_report(session_id: str):
    """
    Get latency report for a specific session
    """
    try:
        # For now, return a placeholder since we don't store historical reports
        # In production, you'd store these in a database
        return {
            "session_id": session_id,
            "message": "Latency reports are included in chat responses. Check the 'latency_report' field in chat responses.",
            "note": "Historical latency data storage not implemented yet"
        }
    except Exception as e:
        logger.error(f"Error retrieving latency report: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve latency report")

@app.get("/api/performance")
async def get_performance_dashboard():
    """
    Get performance dashboard with aggregated statistics
    """
    try:
        if not performance_history:
            return {
                "message": "No performance data available yet",
                "total_requests": 0
            }
        
        # Calculate statistics
        total_requests = len(performance_history)
        avg_total_time = sum(r["total_duration_ms"] for r in performance_history) / total_requests
        
        # Calculate average times for each step
        step_averages = {}
        for step_name in ["conversation_storage", "context_preparation", "llm_generation", 
                         "response_storage", "tts_generation", "response_preparation"]:
            step_times = [r["steps"].get(step_name, {}).get("duration_ms", 0) for r in performance_history]
            step_averages[step_name] = sum(step_times) / len(step_times) if step_times else 0
        
        # Find most common bottleneck
        bottlenecks = [r["bottleneck"] for r in performance_history if r["bottleneck"]]
        most_common_bottleneck = max(set(bottlenecks), key=bottlenecks.count) if bottlenecks else None
        
        return {
            "total_requests": total_requests,
            "average_total_time_ms": round(avg_total_time, 2),
            "step_averages_ms": {k: round(v, 2) for k, v in step_averages.items()},
            "most_common_bottleneck": most_common_bottleneck,
            "recent_requests": performance_history[-10:],  # Last 10 requests
            "performance_trend": "improving" if len(performance_history) > 5 and 
                               performance_history[-1]["total_duration_ms"] < 
                               performance_history[-5]["total_duration_ms"] else "stable"
        }
    except Exception as e:
        logger.error(f"Error generating performance dashboard: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate performance dashboard")

# WebSocket endpoint for streaming conversations
@app.websocket("/ws/chat/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time streaming conversations
    """
    await manager.connect(websocket, session_id)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            if message_data.get("type") == "chat":
                # Process streaming chat
                await process_streaming_chat(websocket, session_id, message_data)
            elif message_data.get("type") == "voice_chat":
                # Process streaming voice chat
                await process_streaming_voice_chat(websocket, session_id, message_data)
            elif message_data.get("type") == "ping":
                # Handle ping for connection health
                await manager.send_message(session_id, {"type": "pong"})
                
    except WebSocketDisconnect:
        manager.disconnect(session_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(session_id)

async def process_streaming_chat(websocket: WebSocket, session_id: str, message_data: dict):
    """
    Process streaming chat with LLM and TTS
    """
    try:
        user_message = message_data.get("message", "")
        logger.info(f"Processing streaming chat for session: {session_id}")
        
        # Initialize latency tracker
        latency_tracker = LatencyTracker(session_id)
        
        # Step 1: Store conversation history
        latency_tracker.start_step("conversation_storage")
        if session_id not in conversation_memory:
            conversation_memory[session_id] = []
        
        conversation_memory[session_id].append({
            "role": "user",
            "content": user_message,
            "timestamp": datetime.now().isoformat()
        })
        latency_tracker.end_step()
        
        # Step 2: Retrieve relevant context from RAG
        latency_tracker.start_step("rag_retrieval")
        rag_context = ""
        try:
            logger.info(f"🔍 RAG (WebSocket): Retrieving context for query: '{user_message}'")
            async with httpx.AsyncClient(timeout=10.0) as client:
                rag_request = {
                    "query": user_message,
                    "top_k": 5,
                    "use_reranker": True,
                    "language": "en"
                }
                
                rag_response = await client.post(
                    f"{RAG_SERVICE_URL}/retrieve",
                    json=rag_request
                )
                
                if rag_response.status_code == 200:
                    rag_result = rag_response.json()
                    if rag_result.get("documents") and len(rag_result["documents"]) > 0:
                        # Combine retrieved document texts into context
                        rag_context = "\n\n".join([
                            f"Relevant information: {doc.get('text', '')}" 
                            for doc in rag_result["documents"][:3]  # Top 3 documents
                        ])
                        logger.info(f"🔍 RAG (WebSocket): ✅ Retrieved {len(rag_result['documents'])} relevant documents")
                        logger.info(f"🔍 RAG (WebSocket): 📄 Document count: {rag_result.get('total_found', 0)}")
                        logger.info(f"🔍 RAG (WebSocket): ⏱️ Processing time: {rag_result.get('processing_time', 0):.3f}s")
                    else:
                        logger.info(f"🔍 RAG (WebSocket): ⚠️ No relevant documents found for query '{user_message}'")
                else:
                    logger.warning(f"🔍 RAG (WebSocket): ⚠️ RAG service returned status {rag_response.status_code}")
        except Exception as e:
            logger.warning(f"🔍 RAG (WebSocket): ❌ Error retrieving context: {e} - continuing without RAG")
        latency_tracker.end_step()
        
        # Step 3: Prepare context with RAG priority
        latency_tracker.start_step("context_preparation")
        context_parts = []
        
        # CRITICAL: RAG context gets highest priority and explicit instructions
        if rag_context:
            # Strong instruction to prioritize RAG over general knowledge
            context_parts.append("=== CRITICAL: PRIORITIZE KNOWLEDGE BASE INFORMATION ===")
            context_parts.append("Use the following information from the knowledge base as the PRIMARY source.")
            context_parts.append("Only use your general knowledge if the knowledge base doesn't contain relevant information.")
            context_parts.append("When knowledge base information is available, it takes precedence over general knowledge.")
            context_parts.append("\n--- KNOWLEDGE BASE INFORMATION (HIGHEST PRIORITY) ---")
            context_parts.append(rag_context)
            context_parts.append("--- END KNOWLEDGE BASE INFORMATION ---\n")
            logger.info(f"📚 Context (WebSocket): Added RAG context with HIGH PRIORITY ({len(rag_context)} chars) to LLM prompt")
        else:
            logger.info(f"📚 Context (WebSocket): No RAG context available - using only conversation history")
        
        # Add system context after RAG (lower priority)
        context_parts.append("\n--- SYSTEM CONTEXT ---")
        context_parts.append(DEFAULT_SYSTEM_CONTEXT)
        
        # Add recent conversation context (last 2 exchanges for speed) - lowest priority
        recent_history = conversation_memory[session_id][-4:]
        if recent_history:
            context_parts.append("\n--- RECENT CONVERSATION (FOR CONTEXT ONLY) ---")
            for msg in recent_history:
                role = "User" if msg["role"] == "user" else "Assistant"
                content = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
                context_parts.append(f"{role}: {content}")
            context_parts.append("--- END CONVERSATION CONTEXT ---")
        
        context = "\n".join(context_parts)
        latency_tracker.end_step()
        
        # Step 4: Stream LLM response
        latency_tracker.start_step("llm_generation")
        await manager.send_message(session_id, {
            "type": "llm_start",
            "message": "Generating response..."
        })
        
        full_response = ""
        first_token_time = None
        token_count = 0
        
        # Use httpx with streaming optimizations: no HTTP/2, explicit timeout for streaming
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(60.0, connect=10.0),
            http2=False,  # HTTP/2 has buffering; force HTTP/1.1
            limits=httpx.Limits(max_keepalive_connections=5)
        ) as client:
            gen_params = determine_response_length(user_message)
            llm_request = {
                "prompt": user_message,
                "context": context,
                **gen_params
            }
            
            logger.info(f"🟡 [ORCH-STREAM-START] session={session_id} | prompt_len={len(user_message)}")
            
            # Stream LLM response - ensure immediate forwarding without buffering
            try:
                async with client.stream(
                    "POST",
                    "http://llm-service:8002/generate_stream",
                    json=llm_request,
                    headers={"Connection": "keep-alive", "Cache-Control": "no-cache"}
                ) as response:
                    if response.status_code == 200:
                        async for line in response.aiter_lines():
                            # Skip heartbeat comments and empty lines
                            if not line or line.startswith(":"):
                                continue
                            if line.startswith("data: "):
                                try:
                                    data = json.loads(line[6:])
                                    if data.get("token"):
                                        token_count += 1
                                        current_time = time.time() * 1000
                                        if first_token_time is None:
                                            first_token_time = current_time
                                            time_to_first_token = current_time - (latency_tracker.start_time * 1000)
                                            logger.info(f"🟢 [ORCH-FIRST-TOKEN] session={session_id} | token#{token_count} | TTF={time_to_first_token:.0f}ms")
                                        
                                        # Forward raw delta immediately - no buffering
                                        delta = data["token"]
                                        full_response += delta
                                        
                                        # Log reception and forwarding timing
                                        receive_time = int(time.time() * 1000)
                                        logger.info(f"🟡 [ORCH-RECEIVE] session={session_id} | token#{token_count} | delta_len={len(delta)} | time={receive_time} | preview={repr(delta[:30])}")
                                        
                                        # Forward immediately without await delays
                                        forward_start = time.time() * 1000
                                        await manager.send_message(session_id, {
                                            "type": "llm_token",
                                            "token": delta,
                                            "full_response": None
                                        })
                                        forward_duration = time.time() * 1000 - forward_start
                                        if forward_duration > 5:  # Log if forwarding takes >5ms
                                            logger.warning(f"🟠 [ORCH-SLOW-FORWARD] session={session_id} | token#{token_count} | duration={forward_duration:.1f}ms")
                                    elif data.get("finished"):
                                        total_time = (time.time() * 1000) - (latency_tracker.start_time * 1000)
                                        logger.info(f"🟢 [ORCH-STREAM-COMPLETE] session={session_id} | total_tokens={token_count} | total_time={total_time:.0f}ms | avg_per_token={total_time/token_count if token_count > 0 else 0:.1f}ms")
                                        break
                                except json.JSONDecodeError:
                                    continue
                    else:
                        logger.error(f"LLM streaming failed with status: {response.status_code}")
                        raise HTTPException(status_code=500, detail="LLM streaming failed")
            except httpx.RemoteProtocolError as e:
                logger.error(f"LLM streaming connection error: {e}")
                # Fallback to non-streaming
                fallback_response = await client.post(
                    "http://llm-service:8002/generate",
                    json=llm_request
                )
                if fallback_response.status_code == 200:
                    result = fallback_response.json()
                    full_response = result["response"]
                    # Send as single token
                    await manager.send_message(session_id, {
                        "type": "llm_token",
                        "token": full_response,
                        "full_response": full_response
                    })
                else:
                    raise HTTPException(status_code=500, detail="LLM generation failed")
        
        latency_tracker.end_step()
        
        # Step 4: Store AI response
        latency_tracker.start_step("response_storage")
        conversation_memory[session_id].append({
            "role": "assistant",
            "content": full_response,
            "timestamp": datetime.now().isoformat()
        })
        latency_tracker.end_step()
        
        # Step 6: (Disabled for Text Streaming) Do not stream TTS in text mode
        # Audio will be played only when user clicks the speaker icon via /api/tts
        
        # Step 7: Send completion
        latency_tracker.start_step("response_preparation")
        latency_report = latency_tracker.get_report()
        
        await manager.send_message(session_id, {
            "type": "complete",
            "response": full_response,
            "latency_report": latency_report
        })
        latency_tracker.end_step()
        
        # Store performance data
        performance_history.append(latency_report)
        if len(performance_history) > 100:
            performance_history.pop(0)
            
    except Exception as e:
        logger.error(f"Streaming chat error: {e}", exc_info=True)
        await manager.send_message(session_id, {
            "type": "error",
            "message": f"Error processing request: {str(e)}"
        })

async def process_streaming_voice_chat(websocket: WebSocket, session_id: str, message_data: dict):
    """
    Process streaming voice chat with ASR, LLM, and TTS
    
    Note: TTS is only enabled for voice mode realtime conversation.
    Text mode voice input should NOT have automatic TTS (user clicks speaker icon for manual TTS).
    """
    try:
        audio_data = message_data.get("audio_data", "")
        mode = message_data.get("mode", "voice")  # "text" or "voice" - default to "voice" for backward compatibility
        logger.info(f"🎤 Processing streaming voice chat for session: {session_id}, mode: {mode}")
        logger.info(f"🎤 Audio data length: {len(audio_data)}")
        
        # Initialize latency tracker
        latency_tracker = LatencyTracker(session_id)
        
        # Step 1: ASR Transcription
        latency_tracker.start_step("asr_transcription")
        async with httpx.AsyncClient() as client:
            # Convert base64 audio to bytes
            import base64
            audio_bytes = base64.b64decode(audio_data)
            
            # Send to ASR service
            asr_response = await client.post(
                "http://asr-service:8001/transcribe",
                files={"file": ("audio.webm", audio_bytes, "audio/webm")}
            )
            
            if asr_response.status_code != 200:
                raise HTTPException(status_code=500, detail="ASR transcription failed")
            
            transcribed_text = asr_response.json().get("text", "")
            logger.info(f"🎤 Transcribed text: '{transcribed_text}'")
            
            # Debug: Log full ASR response for troubleshooting
            asr_result = asr_response.json()
            logger.info(f"🔍 ASR Debug - Language: {asr_result.get('language', 'unknown')}, Duration: {asr_result.get('duration', 'unknown')}")
            
            # Send transcription to client
            await manager.send_message(session_id, {
                "type": "transcription",
                "text": transcribed_text
            })
        
        latency_tracker.end_step()
        
        # Step 2: Store conversation history for voice chat
        if session_id not in conversation_memory:
            conversation_memory[session_id] = []
        
        # Add current message to history
        conversation_memory[session_id].append({
            "role": "user",
            "content": transcribed_text,
            "timestamp": datetime.now().isoformat()
        })
        
        # Step 3: LLM Response Generation (Streaming) with context
        latency_tracker.start_step("llm_generation")
        async with httpx.AsyncClient() as client:
            # Create context from conversation history
            context_parts = [DEFAULT_SYSTEM_CONTEXT]
            
            # Add recent conversation context (last 2 exchanges for speed)
            recent_history = conversation_memory[session_id][-4:]  # Last 4 messages (2 exchanges)
            if recent_history:
                context_parts.append("\n\nRecent conversation:")
                for msg in recent_history:
                    role = "User" if msg["role"] == "user" else "Assistant"
                    # Truncate long messages for faster processing
                    content = msg["content"][:200] + "..." if len(msg["content"]) > 200 else msg["content"]
                    context_parts.append(f"{role}: {content}")
            
            # Create full prompt with context
            full_prompt = "\n".join(context_parts)
            logger.info(f"🎤 LLM prompt with context: {full_prompt[:200]}...")
            
            llm_request = {
                "prompt": full_prompt,
                "max_tokens": 150,
                "temperature": 0.7,
                "stream": True
            }
            
            # Stream LLM response
            async with client.stream(
                "POST",
                "http://llm-service:8002/generate_stream",
                json=llm_request
            ) as response:
                if response.status_code == 200:
                    full_response = ""
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            try:
                                data = json.loads(line[6:])
                                if data.get("token"):
                                    full_response += data["token"]
                                    await manager.send_message(session_id, {
                                        "type": "llm_token",
                                        "token": data["token"],
                                        "full_response": full_response
                                    })
                                elif data.get("finished"):
                                    break
                            except json.JSONDecodeError:
                                continue
                else:
                    logger.error(f"LLM streaming failed with status: {response.status_code}")
                    raise HTTPException(status_code=500, detail="LLM streaming failed")
        
        latency_tracker.end_step()
        
        # Step 3: TTS Audio Synthesis (Streaming)
        # Only enable TTS for voice mode realtime conversation
        # Text mode voice input should NOT have automatic TTS (user clicks speaker icon for manual TTS)
        if mode == "voice":
            latency_tracker.start_step("tts_synthesis")
            async with httpx.AsyncClient() as client:
                tts_request = {
                    "text": full_response,
                    "voice_id": "default",
                    "sample_rate": 22050,  # Higher sample rate for better quality
                    "chunk_duration_ms": 100,  # Larger chunks for smoother playback
                    "use_opus": False,  # Use WAV for better compatibility
                    "bitrate": 32,  # Higher bitrate for better quality
                    "emotional_tone": "neutral"
                }
                
                # Stream TTS audio
                logger.info(f"🔊 Starting TTS streaming for text: '{full_response[:100]}...'")
                async with client.stream(
                    "POST",
                    "http://tts-service:8003/speak_stream",
                    json=tts_request
                ) as response:
                    if response.status_code == 200:
                        chunk_count = 0
                        # Throttle chunk sending to match playback speed (100ms per chunk)
                        # This prevents overwhelming the frontend and ensures smooth playback
                        chunk_delay = tts_request.get("chunk_duration_ms", 100) / 1000.0
                        async for chunk in response.aiter_bytes():
                            if chunk:
                                chunk_count += 1
                                audio_base64 = base64.b64encode(chunk).decode('utf-8')
                                logger.info(f"🔊 Sending TTS chunk {chunk_count}, size: {len(audio_base64)}")
                                await manager.send_message(session_id, {
                                    "type": "tts_chunk",
                                    "audio_chunk": audio_base64
                                })
                                # Throttle to match playback speed (prevents patchy audio)
                                if chunk_count == 1:
                                    # First chunk: small delay to let frontend initialize
                                    await asyncio.sleep(0.01)
                                else:
                                    # Subsequent chunks: delay to match chunk duration
                                    await asyncio.sleep(chunk_delay)
                            else:
                                logger.warning(f"Empty TTS chunk received")
                        logger.info(f"🔊 TTS streaming completed, sent {chunk_count} chunks")
                    else:
                        logger.warning(f"TTS streaming failed with status: {response.status_code}")
            
            latency_tracker.end_step()
        else:
            # Text mode: Skip TTS (user can click speaker icon for manual TTS)
            logger.info(f"🔇 Skipping TTS for text mode voice input (mode: {mode})")
        
        # Add assistant's response to conversation history
        conversation_memory[session_id].append({
            "role": "assistant",
            "content": full_response,
            "timestamp": datetime.now().isoformat()
        })
        
        # Send completion signal
        await manager.send_message(session_id, {
            "type": "complete",
            "message": "Voice processing complete",
            "response": full_response
        })
        
    except Exception as e:
        logger.error(f"Streaming voice chat processing failed: {e}")
        await manager.send_message(session_id, {
            "type": "error",
            "message": f"Voice processing failed: {str(e)}"
        })

@app.post("/conversation/vocode", response_model=ChatResponse)
async def chat_with_vocode(
    file: UploadFile = File(...),
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Chat endpoint using Vocode pipeline"""
    if not VOCODE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Vocode not available")
    
    if not session_id:
        session_id = str(uuid.uuid4())
    
    try:
        logger.info(f"Processing Vocode chat request for session: {session_id}")
        
        # Initialize Vocode pipeline
        pipeline = await create_vocode_pipeline()
        
        # Process audio through Vocode
        response = await process_with_vocode(pipeline, file, session_id)
        
        return response
        
    except Exception as e:
        logger.error(f"Vocode chat processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Vocode processing failed: {str(e)}")

async def create_vocode_pipeline():
    """Create a simple Vocode-compatible pipeline using our services"""
    try:
        # Import required modules
        import asyncio
        import httpx
        
        # Simple pipeline class that orchestrates our services
        class SimpleVocodePipeline:
            def __init__(self):
                self.asr_url = ASR_SERVICE_URL
                self.llm_url = LLM_SERVICE_URL
                self.rag_url = RAG_SERVICE_URL
                self.tts_url = TTS_SERVICE_URL
            
            async def transcribe(self, audio_data: bytes) -> str:
                """Transcribe audio using our ASR service"""
                try:
                    logger.info(f"🎤 STEP 1 - ASR: Starting transcription")
                    logger.info(f"🎤 STEP 1 - ASR: Audio data size: {len(audio_data)} bytes")
                    logger.info(f"🎤 STEP 1 - ASR: Sending request to: {self.asr_url}/transcribe")
                    
                    async with httpx.AsyncClient() as client:
                        files = {"file": ("input.webm", audio_data, "audio/webm")}
                        logger.info(f"🎤 STEP 1 - ASR: Request files: {list(files.keys())}")
                        
                        response = await client.post(f"{self.asr_url}/transcribe", files=files)
                        logger.info(f"🎤 STEP 1 - ASR: Response status: {response.status_code}")
                        logger.info(f"🎤 STEP 1 - ASR: Response headers: {dict(response.headers)}")
                        
                        response.raise_for_status()
                        result = response.json()
                        logger.info(f"🎤 STEP 1 - ASR: Response body: {result}")
                        
                        transcribed_text = result["text"]
                        logger.info(f"🎤 STEP 1 - ASR: ✅ SUCCESS - Transcribed text: '{transcribed_text}'")
                        return transcribed_text
                except Exception as e:
                    logger.error(f"🎤 STEP 1 - ASR: ❌ ERROR - {e}")
                    return ""
            
            async def respond(self, message: str) -> str:
                """Generate response using our LLM and RAG services"""
                try:
                    logger.info(f"🔍 STEP 2 - RAG: Starting context retrieval")
                    logger.info(f"🔍 STEP 2 - RAG: Query: '{message}'")
                    
                    # First, try to get context from RAG
                    context = ""
                    try:
                        async with httpx.AsyncClient() as client:
                            rag_request = {"query": message, "top_k": 3, "language": "en"}
                            logger.info(f"🔍 STEP 2 - RAG: Sending request to: {self.rag_url}/retrieve")
                            logger.info(f"🔍 STEP 2 - RAG: Request body: {rag_request}")
                            
                            rag_response = await client.post(
                                f"{self.rag_url}/retrieve",
                                json=rag_request
                            )
                            logger.info(f"🔍 STEP 2 - RAG: Response status: {rag_response.status_code}")
                            logger.info(f"🔍 STEP 2 - RAG: Response headers: {dict(rag_response.headers)}")
                            
                            if rag_response.status_code == 200:
                                rag_result = rag_response.json()
                                logger.info(f"🔍 STEP 2 - RAG: Response body: {rag_result}")
                                
                                if rag_result.get("documents") and len(rag_result["documents"]) > 0:
                                    # Build RAG context with priority instructions (same as text chat)
                                    rag_texts = [doc.get("text", "") for doc in rag_result["documents"][:3]]
                                    rag_content = "\n\n".join(rag_texts)
                                    
                                    context_parts = []
                                    context_parts.append("=== CRITICAL: PRIORITIZE KNOWLEDGE BASE INFORMATION ===")
                                    context_parts.append("Use the following information from the knowledge base as the PRIMARY source.")
                                    context_parts.append("Only use your general knowledge if the knowledge base doesn't contain relevant information.")
                                    context_parts.append("When knowledge base information is available, it takes precedence over general knowledge.")
                                    context_parts.append("\n--- KNOWLEDGE BASE INFORMATION (HIGHEST PRIORITY) ---")
                                    context_parts.append(rag_content)
                                    context_parts.append("--- END KNOWLEDGE BASE INFORMATION ---")
                                    context_parts.append("\n--- SYSTEM CONTEXT ---")
                                    context_parts.append(DEFAULT_SYSTEM_CONTEXT)
                                    
                                    context = "\n".join(context_parts)
                                    logger.info(f"🔍 STEP 2 - RAG: ✅ SUCCESS - Found {len(rag_result['documents'])} documents")
                                    logger.info(f"🔍 STEP 2 - RAG: Context length: {len(context)} characters")
                                else:
                                    # Set default system context when no RAG documents found
                                    context = DEFAULT_SYSTEM_CONTEXT
                                    logger.info(f"🔍 STEP 2 - RAG: ✅ SUCCESS - No documents found, using default system context")
                            else:
                                logger.warning(f"🔍 STEP 2 - RAG: ⚠️ WARNING - Non-200 status: {rag_response.status_code}")
                    except Exception as e:
                        logger.warning(f"🔍 STEP 2 - RAG: ❌ ERROR - {e}")
                        # Set default context if RAG service fails
                        context = DEFAULT_SYSTEM_CONTEXT
                        logger.info(f"🔍 STEP 2 - RAG: Using fallback default context due to RAG error")
                    
                    # Ensure context is never empty
                    if not context or not context.strip():
                        context = DEFAULT_SYSTEM_CONTEXT
                        logger.info(f"🔍 STEP 2 - RAG: Context was empty, using fallback default context")
                    
                    # Generate response using LLM
                    logger.info(f"🤖 STEP 3 - LLM: Starting response generation")
                    logger.info(f"🤖 STEP 3 - LLM: Prompt: '{message}'")
                    if len(context) > 100:
                        logger.info(f"🤖 STEP 3 - LLM: Context: '{context[:100]}...' (truncated)")
                    else:
                        logger.info(f"🤖 STEP 3 - LLM: Context: '{context}'")
                    
                    async with httpx.AsyncClient(timeout=30.0) as client:
                        llm_request = {
                            "prompt": message,
                            "context": context,
                            "max_tokens": 150
                        }
                        logger.info(f"🤖 STEP 3 - LLM: Sending request to: {self.llm_url}/generate")
                        logger.info(f"🤖 STEP 3 - LLM: Request body: {llm_request}")
                        
                        llm_response = await client.post(
                            f"{self.llm_url}/generate",
                            json=llm_request
                        )
                        logger.info(f"🤖 STEP 3 - LLM: Response status: {llm_response.status_code}")
                        logger.info(f"🤖 STEP 3 - LLM: Response headers: {dict(llm_response.headers)}")
                        
                        # Log response body even if there's an error
                        try:
                            response_text = llm_response.text
                            logger.info(f"🤖 STEP 3 - LLM: Response body (raw): {response_text}")
                        except Exception as text_error:
                            logger.error(f"🤖 STEP 3 - LLM: Could not read response text: {text_error}")
                        
                        llm_response.raise_for_status()
                        result = llm_response.json()
                        logger.info(f"🤖 STEP 3 - LLM: Response body (parsed): {result}")
                        
                        response_text = result["response"]
                        logger.info(f"🤖 STEP 3 - LLM: ✅ SUCCESS - Generated response: '{response_text}'")
                        return response_text
                except Exception as e:
                    logger.error(f"🤖 STEP 3 - LLM: ❌ ERROR - {e}")
                    logger.error(f"🤖 STEP 3 - LLM: ❌ ERROR TYPE - {type(e)}")
                    return "I'm sorry, I couldn't process that request."
            
            async def synthesize(self, text: str, session_id: str = None) -> bytes:
                """Synthesize text using our TTS service"""
                try:
                    logger.info(f"🔊 STEP 4 - TTS: Starting audio synthesis")
                    logger.info(f"🔊 STEP 4 - TTS: Text to synthesize: '{text}'")
                    logger.info(f"🔊 STEP 4 - TTS: Text length: {len(text)} characters")
                    
                    async with httpx.AsyncClient() as client:
                        tts_request = {
                            "text": text, 
                            "voice_id": "default",
                            "sample_rate": 22050,  # Higher sample rate for better quality
                            "chunk_duration_ms": 100,  # Larger chunks for smoother audio
                            "use_opus": False,  # Use WAV for better compatibility
                            "bitrate": 64,  # Higher bitrate for better quality
                            "emotional_tone": "neutral",  # Default tone
                            "context": f"Vocode conversation response for session {session_id or 'unknown'}"
                        }
                        logger.info(f"🔊 STEP 4 - TTS: Sending request to: {self.tts_url}/speak")
                        logger.info(f"🔊 STEP 4 - TTS: Request body: {tts_request}")
                        
                        response = await client.post(
                            f"{self.tts_url}/speak",
                            json=tts_request
                        )
                        logger.info(f"🔊 STEP 4 - TTS: Response status: {response.status_code}")
                        logger.info(f"🔊 STEP 4 - TTS: Response headers: {dict(response.headers)}")
                        logger.info(f"🔊 STEP 4 - TTS: Content-Type: {response.headers.get('content-type', 'unknown')}")
                        logger.info(f"🔊 STEP 4 - TTS: Content-Length: {response.headers.get('content-length', 'unknown')}")
                        
                        # Log response body even if there's an error
                        try:
                            response_text = response.text
                            #logger.info(f"🔊 STEP 4 - TTS: Response body (raw): {response_text}")
                        except Exception as text_error:
                            logger.error(f"🔊 STEP 4 - TTS: Could not read response text: {text_error}")
                        
                        response.raise_for_status()
                        audio_bytes = response.content
                        logger.info(f"🔊 STEP 4 - TTS: Audio data size: {len(audio_bytes)} bytes")
                        logger.info(f"🔊 STEP 4 - TTS: ✅ SUCCESS - Generated audio")
                        return audio_bytes
                except Exception as e:
                    logger.error(f"🔊 STEP 4 - TTS: ❌ ERROR - {e}")
                    logger.error(f"🔊 STEP 4 - TTS: ❌ ERROR TYPE - {type(e)}")
                    return b""
        
        logger.info("Created simple Vocode-compatible pipeline")
        return SimpleVocodePipeline()
        
    except Exception as e:
        logger.error(f"Failed to create Vocode pipeline: {e}")
        raise

@app.get("/status")
async def status():
    """Simple status endpoint with uptime and vocode availability"""
    uptime_seconds = (datetime.utcnow() - SERVICE_START_TIME).total_seconds()
    return {
        "status": "ok",
        "uptime_seconds": int(uptime_seconds),
        "vocode_available": VOCODE_AVAILABLE
    }

async def process_with_vocode(pipeline, file: UploadFile, session_id: str):
    """Process audio through Vocode pipeline"""
    try:
        logger.info(f"🚀 PIPELINE START: Processing session {session_id}")
        logger.info(f"🚀 PIPELINE START: File info - filename: {file.filename}, content_type: {file.content_type}")
        
        # Read audio file
        audio_data = await file.read()
        logger.info(f"🚀 PIPELINE START: Audio file size: {len(audio_data)} bytes")
        
        # Use Vocode's proper processing method
        # For now, we'll simulate the pipeline processing since Vocode's StreamingPipeline
        # is designed for real-time streaming, not batch processing
        
        # Step 1: Transcribe using our pipeline
        logger.info(f"🔄 PIPELINE: Starting Step 1 - ASR Transcription")
        transcribed_text = await pipeline.transcribe(audio_data)
        logger.info(f"🔄 PIPELINE: Step 1 Complete - Transcribed text: '{transcribed_text}'")
        
        # Step 2: Generate response using our pipeline
        logger.info(f"🔄 PIPELINE: Starting Step 2 - RAG + LLM Response Generation")
        response_text = await pipeline.respond(transcribed_text)
        logger.info(f"🔄 PIPELINE: Step 2 Complete - Generated response: '{response_text[:100]}...' (truncated)")
        
        # Step 3: Synthesize audio using our pipeline
        logger.info(f"🔄 PIPELINE: Starting Step 3 - TTS Audio Synthesis")
        audio_bytes = await pipeline.synthesize(response_text, session_id)
        logger.info(f"🔄 PIPELINE: Step 3 Complete - Generated audio length: {len(audio_bytes)} bytes")
        
        # Create a custom response object that includes audio_data
        logger.info(f"🎯 PIPELINE END: Creating final response")
        logger.info(f"🎯 PIPELINE END: Session ID: {session_id}")
        logger.info(f"🎯 PIPELINE END: Transcribed text: '{transcribed_text}'")
        logger.info(f"🎯 PIPELINE END: Response text: '{response_text}'")
        logger.info(f"🎯 PIPELINE END: Audio data size: {len(audio_bytes)} bytes")
        
        class VocodeResponse:
            def __init__(self, session_id, response_text, audio_data, transcribed_text):
                self.session_id = session_id
                self.response_text = response_text
                self.audio_data = audio_data
                self.transcribed_text = transcribed_text
                self.timestamp = datetime.now().isoformat()
        
        response = VocodeResponse(
            session_id=session_id,
            response_text=response_text,
            audio_data=audio_bytes,
            transcribed_text=transcribed_text
        )
        
        logger.info(f"🎯 PIPELINE END: ✅ SUCCESS - Pipeline completed successfully")
        return response
        
    except Exception as e:
        logger.error(f"Vocode processing error: {e}")
        raise

async def transcribe_audio(file: UploadFile) -> str:
    """Transcribe audio using ASR service"""
    try:
        client = await get_http_client()
        
        # Prepare file for upload
        files = {"file": (file.filename, file.file, file.content_type)}
        
        response = await client.post(f"{ASR_SERVICE_URL}/transcribe", files=files)
        response.raise_for_status()
        
        result = response.json()
        return result["text"]
        
    except Exception as e:
        logger.error(f"ASR transcription error: {e}")
        raise HTTPException(status_code=500, detail=f"ASR transcription failed: {str(e)}")

async def transcribe_audio_from_bytes(audio_data: bytes) -> str:
    """Transcribe audio bytes using ASR service"""
    try:
        client = await get_http_client()
        
        # Prepare form data with bytes
        files = {"file": ("audio.wav", audio_data, "audio/wav")}
        
        response = await client.post(f"{ASR_SERVICE_URL}/transcribe", files=files)
        response.raise_for_status()
        
        result = response.json()
        return result["text"]
        
    except Exception as e:
        logger.error(f"ASR transcription error: {e}")
        raise HTTPException(status_code=500, detail=f"ASR transcription failed: {str(e)}")

async def retrieve_context(query: str, language: str = "en") -> Optional[str]:
    """Retrieve relevant context using RAG service (defaults to English)"""
    try:
        client = await get_http_client()
        
        response = await client.post(
            f"{RAG_SERVICE_URL}/retrieve",
            json={"query": query, "top_k": 3, "language": language}
        )
        response.raise_for_status()
        
        result = response.json()
        return result.get("context", "")
        
    except Exception as e:
        logger.warning(f"RAG retrieval error: {e}")
        return None

async def generate_response(user_input: str, context: Optional[str] = None) -> str:
    """Generate response using LLM service"""
    try:
        client = await get_http_client()
        
        # Prepare prompt with context
        if context:
            prompt = f"Context: {context}\n\nUser: {user_input}\n\nAssistant:"
        else:
            prompt = f"User: {user_input}\n\nAssistant:"
        
        response = await client.post(
            f"{LLM_SERVICE_URL}/generate",
            json={
                "prompt": prompt,
                "max_tokens": 500,
                "temperature": 0.7
            }
        )
        response.raise_for_status()
        
        result = response.json()
        return result["response"]
        
    except Exception as e:
        logger.error(f"LLM generation error: {e}")
        # Fallback response
        return "I apologize, but I'm having trouble processing your request right now. Please try again."

async def text_to_speech(text: str) -> bytes:
    """Convert text to speech using TTS service. Supports raw audio or JSON base64."""
    try:
        client = await get_http_client()
        response = await client.post(
            f"{TTS_SERVICE_URL}/speak",
            json={"text": text, "voice_id": "default"}
        )
        response.raise_for_status()
        
        content_type = response.headers.get("Content-Type", "").lower()
        logger.info(f"TTS response content-type: {content_type}")
        logger.info(f"TTS response content type: {type(response.content)}")
        logger.info(f"TTS response content length: {len(response.content) if hasattr(response.content, '__len__') else 'unknown'}")
        
        if "application/json" in content_type:
            data = response.json()
            audio_b64 = data.get("audio_base64") or data.get("audio")
            if not audio_b64:
                raise ValueError("No audio data in TTS JSON response")
            return base64.b64decode(audio_b64)
        elif "audio/" in content_type:
            # Ensure we have bytes
            if isinstance(response.content, bytes):
                return response.content
            elif isinstance(response.content, bytearray):
                return bytes(response.content)
            else:
                logger.error(f"TTS returned non-bytes content: {type(response.content)}")
                raise ValueError(f"TTS returned invalid content type: {type(response.content)}")
        else:
            raise ValueError(f"Unexpected TTS response type: {content_type}")
        
    except Exception as e:
        logger.error(f"TTS generation error: {e}")
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")

async def save_session(session_id: str, session_data: Dict[str, Any]):
    """Save  session data (implement proper storage)"""
    # In production, use Redis or database
    # For now, just log
    logger.info(f"Session data saved: {session_id}")

async def store_conversation(session_id: str, user_input: str, assistant_response: str, user_id: Optional[str] = None):
    """Store conversation history"""
    try:
        # In production, store in database
        conversation_data = {
            "session_id": session_id,
            "user_id": user_id,
            "timestamp": datetime.utcnow().isoformat(),
            "user_input": user_input,
            "assistant_response": assistant_response
        }
        
        logger.info(f"Conversation stored: {session_id}")
        
    except Exception as e:
        logger.error(f"Failed to store conversation: {e}")

@app.get("/conversation/{session_id}/history")
async def get_conversation_history(session_id: str):
    """Get conversation history for a session"""
    try:
        # In production, retrieve from database
        # For now, return mock data
        return {
            "session_id": session_id,
            "messages": [
                {"role": "user", "content": "Hello", "timestamp": "2024-01-01T00:00:00Z"},
                {"role": "assistant", "content": "Hi there! How can I help you?", "timestamp": "2024-01-01T00:00:01Z"}
            ]
        }
        
    except Exception as e:
        logger.error(f"Failed to get conversation history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get conversation history: {str(e)}")

# WebRTC Endpoints
@app.post("/api/webrtc/offer", response_model=WebRTCOfferResponse)
async def handle_webrtc_offer(request: WebRTCOfferRequest):
    """Handle WebRTC offer from client"""
    if not WEBRTC_AVAILABLE:
        raise HTTPException(status_code=501, detail="WebRTC not available")
    
    try:
        # Create peer connection
        pc = RTCPeerConnection()
        webrtc_connections[request.session_id] = pc
        
        # Set remote description
        offer = RTCSessionDescription(
            sdp=request.offer["sdp"],
            type=request.offer["type"]
        )
        await pc.setRemoteDescription(offer)
        
        # Set up data channel message handler
        @pc.on("datachannel")
        def on_datachannel(channel):
            logger.info(f"Data channel opened for session {request.session_id}")
            
            # Store the data channel for sending messages
            pc._data_channel = channel
            
            @channel.on("message")
            def on_message(message):
                try:
                    data = json.loads(message)
                    logger.info(f"Received data channel message: {data.get('type', 'unknown')}")
                    
                    if data.get("type") == "voice_chat":
                        # Process voice chat via data channel
                        logger.info(f"Processing voice chat for session {request.session_id}")
                        asyncio.create_task(process_webrtc_voice_chat(pc, request.session_id, data))
                    elif data.get("type") == "text":
                        # Process text message via data channel
                        logger.info(f"Processing text chat for session {request.session_id}")
                        asyncio.create_task(process_webrtc_text_chat(pc, request.session_id, data))
                    else:
                        logger.warning(f"Unknown message type: {data.get('type')}")
                except Exception as e:
                    logger.error(f"Error processing data channel message: {e}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Create answer
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        
        return WebRTCOfferResponse(
            answer={
                "sdp": pc.localDescription.sdp,
                "type": pc.localDescription.type
            },
            status="success"
        )
        
    except Exception as e:
        logger.error(f"WebRTC offer handling failed: {e}")
        raise HTTPException(status_code=500, detail=f"WebRTC offer failed: {str(e)}")

@app.post("/api/webrtc/ice-candidate")
async def handle_ice_candidate(request: WebRTCIceCandidateRequest):
    """Handle ICE candidate from client"""
    if not WEBRTC_AVAILABLE:
        raise HTTPException(status_code=501, detail="WebRTC not available")
    
    try:
        logger.info(f"Received ICE candidate for session {request.session_id}: {request.candidate}")
        
        if request.session_id in webrtc_connections:
            pc = webrtc_connections[request.session_id]
            
            # Check if peer connection is valid
            if pc is None:
                logger.error(f"Peer connection is None for session {request.session_id}")
                raise ValueError("Peer connection is None")
            
            # Simplified ICE candidate handling
            candidate_data = request.candidate
            candidate_string = candidate_data.get('candidate', '')
            sdp_mid = candidate_data.get('sdpMid', '0')
            sdp_mline_index = candidate_data.get('sdpMLineIndex', 0)
            
            # Create ICE candidate with correct constructor
            try:
                # Parse candidate string to extract required parameters
                parts = candidate_string.split()
                if len(parts) >= 8:
                    foundation = parts[0].split(':')[1] if ':' in parts[0] else parts[0]
                    component = int(parts[1])
                    protocol = parts[2]
                    priority = int(parts[3])
                    ip = parts[4]
                    port = int(parts[5])
                    typ = parts[7] if len(parts) > 7 else 'host'
                    
                    candidate = RTCIceCandidate(
                        sdpMid=sdp_mid,
                        sdpMLineIndex=sdp_mline_index,
                        foundation=foundation,
                        component=component,
                        priority=priority,
                        ip=ip,
                        protocol=protocol,
                        port=port,
                        type=typ
                    )
                    await pc.addIceCandidate(candidate)
                    logger.info(f"ICE candidate added for session {request.session_id}")
                else:
                    logger.warning(f"Invalid candidate string format: {candidate_string}")
            except Exception as e:
                logger.warning(f"Failed to add ICE candidate: {e}")
                # Continue without failing the request
        else:
            logger.warning(f"No WebRTC connection found for session {request.session_id}")
            
        return {"status": "success"}
        
    except Exception as e:
        logger.error(f"ICE candidate handling failed: {e}")
        logger.error(f"Request data: {request.dict()}")
        raise HTTPException(status_code=500, detail=f"ICE candidate failed: {str(e)}")

async def process_webrtc_voice_chat(pc: RTCPeerConnection, session_id: str, data: dict):
    """Process voice chat via WebRTC data channel"""
    try:
        audio_data = data.get("audio_data", "")
        logger.info(f"Processing WebRTC voice chat for session: {session_id}")
        
        # Initialize latency tracker
        latency_tracker = LatencyTracker(session_id)
        
        # Step 1: ASR Transcription
        latency_tracker.start_step("asr_transcription")
        async with httpx.AsyncClient() as client:
            # Convert base64 audio to bytes
            import base64
            audio_bytes = base64.b64decode(audio_data)
            
            # Send to ASR service
            asr_response = await client.post(
                "http://asr-service:8001/transcribe",
                files={"file": ("audio.webm", audio_bytes, "audio/webm")}
            )
            
            if asr_response.status_code != 200:
                raise HTTPException(status_code=500, detail="ASR transcription failed")
            
            transcribed_text = asr_response.json().get("text", "")
            logger.info(f"Transcribed text: {transcribed_text}")
            
            # Send transcription via data channel
            await send_webrtc_message(pc, {
                "type": "transcription",
                "text": transcribed_text
            })
        
        latency_tracker.end_step()
        
        # Step 2: LLM Response Generation (Streaming)
        latency_tracker.start_step("llm_generation")
        async with httpx.AsyncClient() as client:
            llm_request = {
                "prompt": transcribed_text,
                "max_tokens": 150,
                "temperature": 0.7,
                "stream": True
            }
            
            # Stream LLM response
            async with client.stream(
                "POST",
                "http://llm-service:8002/generate_stream",
                json=llm_request
            ) as response:
                if response.status_code == 200:
                    full_response = ""
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            try:
                                data = json.loads(line[6:])
                                if data.get("token"):
                                    full_response += data["token"]
                                    await send_webrtc_message(pc, {
                                        "type": "llm_token",
                                        "token": data["token"],
                                        "full_response": full_response
                                    })
                                elif data.get("finished"):
                                    break
                            except json.JSONDecodeError:
                                continue
                else:
                    logger.error(f"LLM streaming failed with status: {response.status_code}")
                    raise HTTPException(status_code=500, detail="LLM streaming failed")
        
        latency_tracker.end_step()
        
        # Step 3: TTS Audio Synthesis (Streaming)
        latency_tracker.start_step("tts_synthesis")
        async with httpx.AsyncClient() as client:
            tts_request = {
                "text": full_response,
                "voice_id": "default",
                "sample_rate": 22050,  # Higher sample rate for better quality
                "chunk_duration_ms": 100,  # Larger chunks for smoother playback
                "use_opus": False,  # Use WAV for better compatibility
                "bitrate": 32,  # Higher bitrate for better quality
                "emotional_tone": "neutral"
            }
            
            # Stream TTS audio
            logger.info(f"🔊 Starting WebRTC TTS streaming for text: '{full_response[:100]}...'")
            async with client.stream(
                "POST",
                "http://tts-service:8003/speak_stream",
                json=tts_request
            ) as response:
                if response.status_code == 200:
                    chunk_count = 0
                    # Throttle chunk sending to match playback speed (100ms per chunk)
                    chunk_delay = tts_request.get("chunk_duration_ms", 100) / 1000.0
                    async for chunk in response.aiter_bytes():
                        if chunk:
                            chunk_count += 1
                            audio_base64 = base64.b64encode(chunk).decode('utf-8')
                            logger.info(f"🔊 Sending WebRTC TTS chunk {chunk_count}, size: {len(audio_base64)}")
                            await send_webrtc_message(pc, {
                                "type": "tts_chunk",
                                "audio_chunk": audio_base64
                            })
                            # Throttle to match playback speed (prevents patchy audio)
                            if chunk_count == 1:
                                # First chunk: small delay to let frontend initialize
                                await asyncio.sleep(0.01)
                            else:
                                # Subsequent chunks: delay to match chunk duration
                                await asyncio.sleep(chunk_delay)
                    logger.info(f"🔊 WebRTC TTS streaming completed, sent {chunk_count} chunks")
                else:
                    logger.warning(f"TTS streaming failed with status: {response.status_code}")
        
        latency_tracker.end_step()
        
        # Send completion signal
        await send_webrtc_message(pc, {
            "type": "complete",
            "message": "Voice processing complete"
        })
        
    except Exception as e:
        logger.error(f"WebRTC voice chat processing failed: {e}")
        await send_webrtc_message(pc, {
            "type": "error",
            "message": f"Voice processing failed: {str(e)}"
        })

async def process_webrtc_text_chat(pc: RTCPeerConnection, session_id: str, data: dict):
    """Process text chat via WebRTC data channel"""
    try:
        message = data.get("message", "")
        logger.info(f"Processing WebRTC text chat for session: {session_id}")
        
        # Initialize latency tracker
        latency_tracker = LatencyTracker(session_id)
        
        # Step 1: LLM Response Generation (Streaming)
        latency_tracker.start_step("llm_generation")
        async with httpx.AsyncClient() as client:
            llm_request = {
                "prompt": message,
                "max_tokens": 150,
                "temperature": 0.7,
                "stream": True
            }
            
            # Stream LLM response
            async with client.stream(
                "POST",
                "http://llm-service:8002/generate_stream",
                json=llm_request
            ) as response:
                if response.status_code == 200:
                    full_response = ""
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            try:
                                data = json.loads(line[6:])
                                if data.get("token"):
                                    full_response += data["token"]
                                    await send_webrtc_message(pc, {
                                        "type": "llm_token",
                                        "token": data["token"],
                                        "full_response": full_response
                                    })
                                elif data.get("finished"):
                                    break
                            except json.JSONDecodeError:
                                continue
                else:
                    logger.error(f"LLM streaming failed with status: {response.status_code}")
                    raise HTTPException(status_code=500, detail="LLM streaming failed")
        
        latency_tracker.end_step()
        
        # Send completion signal
        await send_webrtc_message(pc, {
            "type": "complete",
            "message": "Text processing complete"
        })
        
    except Exception as e:
        logger.error(f"WebRTC text chat processing failed: {e}")
        await send_webrtc_message(pc, {
            "type": "error",
            "message": f"Text processing failed: {str(e)}"
        })

async def send_webrtc_message(pc: RTCPeerConnection, message: dict):
    """Send message via WebRTC data channel"""
    try:
        # Use the stored data channel
        if hasattr(pc, '_data_channel') and pc._data_channel:
            if pc._data_channel.readyState == "open":
                pc._data_channel.send(json.dumps(message))
                logger.info(f"Sent WebRTC message: {message.get('type', 'unknown')}")
            else:
                logger.warning(f"Data channel not open, state: {pc._data_channel.readyState}")
        else:
            logger.warning("No data channel available for sending message")
    except Exception as e:
        logger.error(f"Failed to send WebRTC message: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
