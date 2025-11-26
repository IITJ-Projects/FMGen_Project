#!/usr/bin/env python3
"""
LLM Service - Language Model Inference Microservice
Uses vLLM for high-throughput inference with LLaMA-3-8B
"""

import os
import logging
import asyncio
import time
import uuid
from typing import Optional, List, Dict, Any, AsyncGenerator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
import torch
from vllm import LLM, SamplingParams
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import EngineArgs

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global vLLM model instances
llm_model = None  # for non-streaming
async_engine: AsyncLLMEngine | None = None  # for true streaming

class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Input prompt for generation")
    context: Optional[str] = Field(None, description="Optional context/retrieved documents")
    max_tokens: int = Field(50, ge=1, le=2048, description="Maximum tokens to generate")
    temperature: float = Field(0.9, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(0.98, ge=0.0, le=1.0, description="Top-p sampling parameter")
    top_k: int = Field(10, ge=1, le=100, description="Top-k sampling parameter")
    stream: bool = Field(False, description="Enable streaming response")
    stop: Optional[List[str]] = Field(None, description="Stop sequences")

class GenerateResponse(BaseModel):
    response: str
    tokens_generated: int
    model_name: str
    processing_time: float

class StreamResponse(BaseModel):
    token: str
    finished: bool
    tokens_generated: int

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    async_engine_loaded: bool
    model_name: str
    max_model_len: int
    gpu_memory_utilization: float

def load_model():
    """Load the vLLM model"""
    global llm_model
    try:
        model_name = os.getenv("MODEL_NAME", "/app/llama3-model/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2")
        max_model_len = int(os.getenv("MAX_MODEL_LEN", "2048"))  # Balanced for performance and quality
        gpu_memory_utilization = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.95"))  # Increased for better performance
        quantization = os.getenv("QUANTIZATION", "awq")
        hf_token = os.getenv("HF_TOKEN")
        
        logger.info(f"Loading vLLM model: {model_name}")
        logger.info(f"Max model length: {max_model_len}")
        logger.info(f"GPU memory utilization: {gpu_memory_utilization}")
        logger.info(f"Quantization: {quantization}")
        logger.info(f"HF Token provided: {'Yes' if hf_token else 'No'}")
        if hf_token:
            logger.info(f"HF Token length: {len(hf_token)}")
        
        # Set HuggingFace token if provided
        if hf_token:
            os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
            logger.info("HuggingFace token configured")
            
            # Also try to login with huggingface-cli
            try:
                import subprocess
                result = subprocess.run(
                    ["huggingface-cli", "login", "--token", hf_token],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                if result.returncode == 0:
                    logger.info("Successfully logged in with huggingface-cli")
                else:
                    logger.warning(f"huggingface-cli login failed: {result.stderr}")
            except Exception as e:
                logger.warning(f"huggingface-cli login error: {e}")
        
        # Try to download model first if token is provided
        if hf_token:
            try:
                from transformers import AutoTokenizer, AutoModelForCausalLM
                logger.info("Attempting to download model with transformers...")
                
                # Download tokenizer
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    token=hf_token,
                    cache_dir="/app/models",
                    trust_remote_code=True
                )
                logger.info("Tokenizer downloaded successfully")
                
                # Download model
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    token=hf_token,
                    cache_dir="/app/models",
                    trust_remote_code=True,
                    torch_dtype="auto",
                    device_map="auto"
                )
                logger.info("Model downloaded successfully")
                
            except Exception as e:
                logger.warning(f"Manual model download failed: {e}")
        
        # Initialize vLLM model with performance optimizations (vLLM 0.5+)
        model_kwargs = {
            "model": model_name,
            "max_model_len": max_model_len,
            "gpu_memory_utilization": gpu_memory_utilization,
            "trust_remote_code": True,
            "download_dir": "/app/models",
            "enforce_eager": True,  # Faster for short sequences
            "max_num_batched_tokens": 1024,  # Reduced for faster batching
            "max_num_seqs": 4,  # Reduced for faster processing
            "block_size": 16,  # Smaller block size for faster processing
            "swap_space": 0,  # Disable swap for faster memory access
            "disable_custom_all_reduce": True,  # Performance optimization
            "cpu_offload_gb": 0  # Disable CPU offload for faster GPU processing
        }
        
        # Add quantization if specified
        if quantization != "none":
            model_kwargs["quantization"] = quantization
        
        # IMPORTANT: Only load AsyncLLMEngine (not LLM) to avoid loading model twice
        # AsyncLLMEngine can handle both streaming and non-streaming requests
        # Loading both LLM and AsyncLLMEngine causes GPU memory exhaustion
        logger.info("Loading model as AsyncLLMEngine only (handles both streaming and non-streaming)...")
        
        # Initialize async engine for streaming (REQUIRED for true streaming)
        # Note: In vLLM 0.11.0, AsyncLLMEngine uses EngineArgs which must be created carefully
        try:
            logger.info("Initializing AsyncLLMEngine for true token-by-token streaming...")
            logger.info(f"Model path: {model_name}")
            logger.info(f"Max model len: {max_model_len}")
            logger.info(f"GPU memory utilization: {gpu_memory_utilization}")
            logger.info(f"Quantization: {quantization}")
            
            # Create EngineArgs with essential parameters
            # Note: EngineArgs may not support all LLM() optimization parameters
            engine_args_kwargs = {
                "model": model_name,
                "max_model_len": max_model_len,
                "gpu_memory_utilization": gpu_memory_utilization,
                "trust_remote_code": True,
                "download_dir": "/app/models",
                "enable_log_requests": False,  # vLLM 0.11.0 may require this attribute
            }
            
            # Add quantization only if specified (don't pass None)
            if quantization and quantization != "none":
                engine_args_kwargs["quantization"] = quantization
            
            logger.info("Creating EngineArgs...")
            try:
                engine_args = EngineArgs(**engine_args_kwargs)
            except TypeError as te:
                # If enable_log_requests is not accepted, remove it and try again
                logger.warning(f"EngineArgs doesn't accept enable_log_requests, retrying without it: {te}")
                engine_args_kwargs.pop("enable_log_requests", None)
                engine_args = EngineArgs(**engine_args_kwargs)
            
            # vLLM 0.11.0 may require enable_log_requests attribute on EngineArgs
            # Set it if it doesn't exist (from_engine_args accesses it)
            if not hasattr(engine_args, 'enable_log_requests'):
                logger.info("Setting enable_log_requests attribute on EngineArgs (vLLM 0.11.0 requirement)")
                engine_args.enable_log_requests = False
            
            # Initialize AsyncLLMEngine from EngineArgs
            logger.info("Initializing AsyncLLMEngine from EngineArgs...")
            global async_engine, llm_model
            async_engine = AsyncLLMEngine.from_engine_args(engine_args)
            logger.info("✅ AsyncLLMEngine initialized successfully for true streaming!")
            
            # Set llm_model to None since we're only using AsyncLLMEngine
            # This prevents confusion - async_engine handles both streaming and non-streaming
            llm_model = None
            logger.info("ℹ️  Using AsyncLLMEngine for both streaming and non-streaming requests")
            
        except Exception as e:
            import traceback
            logger.error(f"❌ CRITICAL: Failed to initialize AsyncLLMEngine!")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error message: {str(e)}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            
            # Fallback: Load LLM if AsyncLLMEngine fails
            logger.warning("Falling back to LLM (non-streaming only)...")
            try:
                llm_model = LLM(**model_kwargs)
                logger.info("✅ LLM model loaded successfully (non-streaming only)")
                async_engine = None
            except Exception as e2:
                logger.error(f"❌ Failed to load LLM as fallback: {e2}")
                llm_model = None
                async_engine = None
        
        logger.info("vLLM model loading completed")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load vLLM model: {e}")
        return False

def format_prompt(user_input: str, context: Optional[str] = None) -> str:
    """
    Format prompt for LLaMA-3-8B-Instruct - generic version
    RAG context is prioritized through explicit instructions in the system context
    """
    
    if context and context.strip():
        # Include context in the system prompt with emphasis on RAG priority
        # The context already contains priority instructions from orchestration service
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{context}

<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    else:
        # Simple user-assistant format without system context
        prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    
    return prompt

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting LLM Service...")
    success = load_model()
    if not success:
        logger.error("Failed to load model during startup")
    
    yield
    
    # Shutdown
    logger.info("Shutting down LLM Service...")
    global llm_model
    if llm_model:
        del llm_model
        llm_model = None

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="LLM Service",
    description="Language Model Inference Service using vLLM",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    # Service is healthy if either async_engine or llm_model is loaded
    is_healthy = (async_engine is not None) or (llm_model is not None)
    return HealthResponse(
        status="healthy" if is_healthy else "unhealthy",
        model_loaded=llm_model is not None,
        async_engine_loaded=async_engine is not None,
        model_name=os.getenv("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.2"),
        max_model_len=int(os.getenv("MAX_MODEL_LEN", "2048")),
        gpu_memory_utilization=float(os.getenv("GPU_MEMORY_UTILIZATION", "0.9"))
    )

@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    """
    Generate text response using vLLM (non-streaming)
    Uses AsyncLLMEngine which handles both streaming and non-streaming
    """
    if async_engine is None and llm_model is None:
        raise HTTPException(status_code=503, detail="LLM model not loaded")
    
    try:
        start_time = time.time()
        
        # Format prompt
        formatted_prompt = format_prompt(request.prompt, request.context)
        
        # Set sampling parameters
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            max_tokens=request.max_tokens,
            stop=request.stop
        )
        
        logger.info(f"Generating response for prompt: {request.prompt[:100]}...")
        
        # Use AsyncLLMEngine if available (preferred), otherwise fallback to LLM
        if async_engine is not None:
            # Use AsyncLLMEngine for non-streaming by collecting all tokens
            # vLLM 0.11.0 API: generate(prompt, sampling_params, request_id) asynchronously yields tokens
            request_id = str(uuid.uuid4())
            generated_text = ""
            tokens_generated = 0
            async for request_output in async_engine.generate(
                formatted_prompt,
                sampling_params,
                request_id
            ):
                output = request_output.outputs[0]
                if output.text:
                    generated_text = output.text  # Get latest full text
                if hasattr(output, 'token_ids') and output.token_ids:
                    tokens_generated = len(output.token_ids)
                if request_output.finished:
                    break
        else:
            # Fallback to LLM for non-streaming
            outputs = llm_model.generate([formatted_prompt], sampling_params)
            generated_text = outputs[0].outputs[0].text.strip()
            tokens_generated = len(outputs[0].outputs[0].token_ids)
        
        processing_time = time.time() - start_time
        
        logger.info(f"Generated {tokens_generated} tokens in {processing_time:.2f}s")
        
        return GenerateResponse(
            response=generated_text.strip(),
            tokens_generated=tokens_generated,
            model_name=os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct"),
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Text generation failed: {str(e)}")

@app.post("/generate_stream")
async def generate_text_stream(request: GenerateRequest):
    """
    Stream generated text tokens
    """
    if async_engine is None and llm_model is None:
        raise HTTPException(status_code=503, detail="LLM model not loaded")
    
    try:
        # Format prompt
        formatted_prompt = format_prompt(request.prompt, request.context)
        
        # Set sampling parameters
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            max_tokens=request.max_tokens,
            stop=request.stop
        )
        
        logger.info(f"Streaming response for prompt: {request.prompt[:100]}...")
        logger.info(f"AsyncLLMEngine status: {'✅ AVAILABLE' if async_engine is not None else '❌ NOT INITIALIZED'}")
        
        async def sse_stream():
            """True streaming using AsyncLLMEngine.generate().
            The generate() method asynchronously yields output tokens as they are generated."""
            try:
                tokens_generated = 0
                last_len = 0
                if async_engine is None:
                    logger.warning("⚠️  AsyncLLMEngine is None - falling back to one-shot generation (NOT true streaming)")
                    raise RuntimeError("AsyncLLMEngine not initialized")
                # vLLM 0.11.0 API: generate(prompt, sampling_params, request_id) asynchronously yields tokens
                request_id = str(uuid.uuid4())
                async for request_output in async_engine.generate(
                    formatted_prompt,
                    sampling_params,
                    request_id
                ):
                    try:
                        output = request_output.outputs[0]
                        current_text = output.text or ""
                        if len(current_text) > last_len:
                            delta = current_text[last_len:]
                            last_len = len(current_text)
                            if delta:
                                tokens_generated += 1
                                timestamp_ms = int(time.time() * 1000)
                                # Log streaming emission for debugging
                                logger.info(f"🔵 [STREAM-OUT] token#{tokens_generated} | delta_len={len(delta)} | time={timestamp_ms} | preview={repr(delta[:50])}")
                                # Yield data line and flush marker for immediate transmission
                                yield f"data: {StreamResponse(token=delta, finished=False, tokens_generated=tokens_generated).model_dump_json()}\n\n"
                                # Empty comment line forces TCP flush (SSE spec)
                                yield ": heartbeat\n\n"
                    except Exception as inner_e:
                        logger.warning(f"Stream delta error: {inner_e}")
                        continue
                # Completion
                timestamp_ms = int(time.time() * 1000)
                logger.info(f"🔵 [STREAM-COMPLETE] total_tokens={tokens_generated} | time={timestamp_ms} | total_length={last_len}")
                yield f"data: {StreamResponse(token='', finished=True, tokens_generated=tokens_generated).model_dump_json()}\n\n"
            except Exception as e:
                logger.warning(f"Async streaming failed: {e}")
                # No fallback - AsyncLLMEngine is the only engine loaded
                # Return error response
                logger.error(f"Cannot fallback: llm_model is None (only AsyncLLMEngine loaded)")
                yield f"data: {StreamResponse(token='', finished=True, tokens_generated=0).model_dump_json()}\n\n"
        
        return StreamingResponse(
            sse_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
                "Transfer-Encoding": "chunked"  # Enable chunked transfer
            }
        )
        
    except Exception as e:
        logger.error(f"Streaming generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Streaming generation failed: {str(e)}")

@app.post("/chat")
async def chat_completion(request: GenerateRequest):
    """
    Chat completion endpoint (compatible with OpenAI format)
    Uses AsyncLLMEngine which handles both streaming and non-streaming
    """
    if async_engine is None and llm_model is None:
        raise HTTPException(status_code=503, detail="LLM model not loaded")
    
    try:
        # Format prompt for chat
        formatted_prompt = format_prompt(request.prompt, request.context)
        
        # Set sampling parameters
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            max_tokens=request.max_tokens,
            stop=request.stop
        )
        
        # Use AsyncLLMEngine if available (preferred), otherwise fallback to LLM
        if async_engine is not None:
            # Use AsyncLLMEngine for non-streaming by collecting all tokens
            # vLLM 0.11.0 API: generate(prompt, sampling_params, request_id) asynchronously yields tokens
            request_id = str(uuid.uuid4())
            generated_text = ""
            prompt_token_count = 0
            completion_token_count = 0
            async for request_output in async_engine.generate(
                formatted_prompt,
                sampling_params,
                request_id
            ):
                output = request_output.outputs[0]
                if output.text:
                    generated_text = output.text  # Get latest full text
                if hasattr(output, 'token_ids') and output.token_ids:
                    completion_token_count = len(output.token_ids)
                if hasattr(request_output, 'prompt_token_ids') and request_output.prompt_token_ids:
                    prompt_token_count = len(request_output.prompt_token_ids)
                if request_output.finished:
                    break
        else:
            # Fallback to LLM for non-streaming
            outputs = llm_model.generate([formatted_prompt], sampling_params)
            generated_text = outputs[0].outputs[0].text.strip()
            prompt_token_count = len(outputs[0].prompt_token_ids)
            completion_token_count = len(outputs[0].outputs[0].token_ids)
        
        # Return OpenAI-compatible format
        return {
            "id": str(uuid.uuid4()),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": os.getenv("MODEL_NAME", "meta-llama/Llama-3-8B-Instruct"),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": generated_text.strip()
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": prompt_token_count,
                "completion_tokens": completion_token_count,
                "total_tokens": prompt_token_count + completion_token_count
            }
        }
        
    except Exception as e:
        logger.error(f"Chat completion error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat completion failed: {str(e)}")

@app.get("/models")
async def list_models():
    """List available models"""
    return {
        "models": [
            {
                "id": os.getenv("MODEL_NAME", "meta-llama/Llama-3-8B-Instruct"),
                "object": "model",
                "created": 0,
                "owned_by": "zevo-ai"
            }
        ]
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8002,
        reload=False,
        log_level="info"
    )
