#!/usr/bin/env python3
"""
RAG Ingestion Application
Handles file upload, text extraction, chunking, and embedding generation
"""

import os
import logging
import asyncio
import uuid
import time
import signal
import re
from typing import List, Dict, Any, Optional
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel
import httpx
import aiofiles

# Text processing imports
import PyPDF2
from docx import Document
import re
from nltk.tokenize import sent_tokenize
import nltk

# LangChain for proven text chunking (industry standard, battle-tested)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Health check cache to reduce frequency of calls
_health_check_cache = {}
_health_check_cache_ttl = 60  # Cache for 60 seconds

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
    title="RAG Ingestion Application",
    description="File upload, text extraction, chunking, and embedding generation for RAG",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add middleware to suppress health check logging
app.add_middleware(SuppressHealthLogMiddleware)

# Configuration
UPLOAD_DIR = Path("uploads")

# Production-ready chunking strategy for LLaMA-3-8B + Qdrant + Dual Embeddings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1024"))  # 1024 chars = ~150-200 words (optimal for LLaMA-3-8B)
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "128"))  # 12.5% overlap for context preservation
MIN_CHUNK_SIZE = int(os.getenv("MIN_CHUNK_SIZE", "256"))  # Minimum chunk size to avoid noise
MAX_CHUNK_SIZE = int(os.getenv("MAX_CHUNK_SIZE", "2048"))  # Maximum chunk size for embedding models

# Advanced chunking settings
SENTENCE_OVERLAP_RATIO = float(os.getenv("SENTENCE_OVERLAP_RATIO", "0.3"))  # 30% sentence overlap
PARAGRAPH_BREAK_WEIGHT = float(os.getenv("PARAGRAPH_BREAK_WEIGHT", "0.8"))  # Prefer paragraph breaks
HEADER_DETECTION = os.getenv("HEADER_DETECTION", "true").lower() == "true"  # Detect document headers
TABLE_HANDLING = os.getenv("TABLE_HANDLING", "preserve").lower()  # How to handle tables

RAG_SERVICE_URL = os.getenv("RAG_SERVICE_URL", "http://localhost:8004")
LLM_SERVICE_URL = os.getenv("LLM_SERVICE_URL", "http://localhost:8002")

# File cleanup configuration
FILE_RETENTION_HOURS = int(os.getenv("FILE_RETENTION_HOURS", "24"))  # Keep files for 24 hours by default
CLEANUP_INTERVAL_HOURS = int(os.getenv("CLEANUP_INTERVAL_HOURS", "6"))  # Run cleanup every 6 hours
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "50"))  # Max file size in MB

# Create upload directory with proper permissions
UPLOAD_DIR.mkdir(exist_ok=True, mode=0o755)
# Ensure directory is writable
try:
    os.chmod(UPLOAD_DIR, 0o755)
except Exception as e:
    logger.warning(f"Could not set permissions on upload directory: {e}")

# Setup templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Pydantic models
class UploadResponse(BaseModel):
    file_id: str
    filename: str
    status: str
    message: str

class ProcessingResponse(BaseModel):
    file_id: str
    chunks_created: int
    embeddings_generated: int
    processing_time: float
    status: str

class ChunkInfo(BaseModel):
    chunk_id: str
    text: str
    metadata: Dict[str, Any]

class HealthResponse(BaseModel):
    status: str
    rag_service: str
    llm_service: str

# Global HTTP client
http_client = None

# File tracking for cleanup
uploaded_files = {}  # file_id -> upload_time

def timeout_handler(signum, frame):
    """Handle timeout signal"""
    raise TimeoutError("Operation timed out")

def safe_chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP, timeout: int = 30) -> List[str]:
    """Safely chunk text with timeout protection"""
    try:
        # Set timeout signal
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
        
        # Perform chunking
        result = chunk_text(text, chunk_size, overlap)
        
        # Clear timeout
        signal.alarm(0)
        return result
        
    except TimeoutError:
        logger.warning(f"Chunking timed out after {timeout} seconds, using simple chunking")
        # Fall back to simple chunking
        return simple_chunk_text(text, chunk_size, overlap)
    except Exception as e:
        logger.warning(f"Chunking failed: {e}, using simple chunking")
        return simple_chunk_text(text, chunk_size, overlap)
    finally:
        # Ensure timeout is cleared
        signal.alarm(0)

def simple_chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Simple chunking without sentence boundary detection"""
    if not text or not text.strip():
        return []
    
    text = text.strip()
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    max_iterations = (len(text) // (chunk_size - overlap)) + 10  # Safety limit
    
    iteration = 0
    while start < len(text) and iteration < max_iterations:
        iteration += 1
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        
        if chunk:
            chunks.append(chunk)
        
        # Move forward with overlap
        new_start = end - overlap
        if new_start <= start:
            # Prevent infinite loop - if we're not moving forward, force increment
            new_start = start + 1
        
        start = new_start
        
        # Safety check
        if start >= len(text) or start >= end:
            break
    
    # If we hit max iterations, just split the remaining text
    if iteration >= max_iterations and start < len(text):
        remaining = text[start:].strip()
        if remaining:
            chunks.append(remaining)
    
    return chunks if chunks else [text]

async def get_http_client():
    """Get HTTP client"""
    global http_client
    if http_client is None:
        http_client = httpx.AsyncClient(timeout=30.0)
    return http_client

def cleanup_old_files():
    """Remove files older than FILE_RETENTION_HOURS"""
    try:
        current_time = time.time()
        cutoff_time = current_time - (FILE_RETENTION_HOURS * 3600)
        
        files_removed = 0
        total_size_freed = 0
        
        for file_path in UPLOAD_DIR.glob("*"):
            if file_path.is_file():
                file_age = current_time - file_path.stat().st_mtime
                
                if file_age > (FILE_RETENTION_HOURS * 3600):
                    try:
                        file_size = file_path.stat().st_size
                        file_path.unlink()
                        files_removed += 1
                        total_size_freed += file_size
                        logger.info(f"Cleaned up old file: {file_path.name}")
                    except Exception as e:
                        logger.error(f"Failed to remove old file {file_path}: {e}")
        
        if files_removed > 0:
            logger.info(f"Cleanup completed: {files_removed} files removed, {total_size_freed / (1024*1024):.2f} MB freed")
            
    except Exception as e:
        logger.error(f"Error during file cleanup: {e}")

async def periodic_cleanup():
    """Run periodic file cleanup"""
    while True:
        try:
            cleanup_old_files()
            await asyncio.sleep(CLEANUP_INTERVAL_HOURS * 3600)  # Sleep for cleanup interval
        except Exception as e:
            logger.error(f"Error in periodic cleanup: {e}")
            await asyncio.sleep(3600)  # Sleep for 1 hour on error

def extract_text_from_pdf(file_path: Path) -> str:
    """Extract text from PDF file"""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting text from PDF {file_path}: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to extract text from PDF: {str(e)}")

def extract_text_from_docx(file_path: Path) -> str:
    """Extract text from DOCX file"""
    try:
        doc = Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting text from DOCX {file_path}: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to extract text from DOCX: {str(e)}")

def extract_text_from_txt(file_path: Path) -> str:
    """Extract text from TXT file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read().strip()
    except Exception as e:
        logger.error(f"Error reading TXT file {file_path}: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to read TXT file: {str(e)}")

def extract_text_from_file(file_path: Path) -> str:
    """Extract text from file based on extension"""
    file_extension = file_path.suffix.lower()
    
    if file_extension == '.pdf':
        return extract_text_from_pdf(file_path)
    elif file_extension == '.docx':
        return extract_text_from_docx(file_path)
    elif file_extension == '.txt':
        return extract_text_from_txt(file_path)
    elif file_extension == '.doc':
        # For .doc files, we'll need to convert them first
        # For now, raise an error
        raise HTTPException(status_code=400, detail="DOC files are not supported yet. Please convert to DOCX.")
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_extension}")

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Production-ready text chunking optimized for LLaMA-3-8B + Qdrant + Dual Embeddings
    
    Strategy:
    - Optimal chunk size: 1024 chars (~150-200 words) for LLaMA-3-8B context window
    - 12.5% overlap for context preservation across chunks
    - Semantic boundary detection (paragraphs > sentences > words)
    - Header and structure preservation
    - Table and list handling
    - Quality filtering for embedding models
    """
    
    # Input validation and sanitization
    if not text or not text.strip():
        return []
    
    # Clean and normalize text
    text = text.strip()
    
    # Handle very short text - return immediately for small texts
    if len(text) <= chunk_size:
        return [text]
    
    # For very small texts (< 5000 chars), skip preprocessing to save time
    # Pre-processing regex can be slow for some patterns
    if len(text) < 5000:
        # Skip preprocessing for small texts - direct chunking is faster
        pass
    else:
        # Pre-process text for better chunking (only for larger texts)
        text = preprocess_text_for_chunking(text)
    
    chunks = []
    start = 0
    
    # For smaller texts, use simpler chunking (faster)
    # For larger texts, use semantic chunking
    use_semantic_chunking = len(text) > 20000  # Only use semantic for large texts
    
    while start < len(text):
        end = start + chunk_size
        
        # Ensure we don't go beyond text length
        if end > len(text):
            end = len(text)
        
        # Find optimal break point using semantic boundaries (only for large texts)
        if use_semantic_chunking and start > 0 and end < len(text):
            optimal_end = find_semantic_boundary(text, start, end, overlap)
            if optimal_end and optimal_end > start and optimal_end <= end:
                end = optimal_end
        elif end < len(text):
            # For smaller texts, just find nearest space/newline (fast)
            # Look backwards from end for a good break point
            search_end = max(end - 100, start + chunk_size - 200)
            for i in range(end - 1, search_end, -1):
                if text[i] in ['\n', '.', '!', '?', ' ']:
                    end = i + 1
                    break
        
        # Extract the chunk
        chunk = text[start:end].strip()
        
        # Quality check: ensure chunk meets minimum requirements
        # For small texts, skip quality check to speed up processing
        if len(text) < 5000 or is_quality_chunk(chunk):
            chunks.append(chunk)
        
        # Move to next position with overlap
        start = end - overlap
        
        # Safety checks
        if start >= len(text) or start >= end:
            break
    
    # Post-processing: merge small chunks and split large ones
    # Skip post-processing for small texts to avoid timeout
    if len(text) < 5000 and chunks:
        # For small texts, return chunks as-is (faster)
        return chunks
    
    final_chunks = post_process_chunks(chunks)
    
    return final_chunks if final_chunks else [text]

def preprocess_text_for_chunking(text: str) -> str:
    """Preprocess text to improve chunking quality"""
    import re
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Preserve paragraph breaks
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    # Handle headers (lines with only uppercase or numbers)
    if HEADER_DETECTION:
        text = re.sub(r'^([A-Z0-9\s]{3,})$', r'HEADER: \1', text, flags=re.MULTILINE)
    
    # Preserve table structures
    if TABLE_HANDLING == "preserve":
        # Mark table boundaries
        text = re.sub(r'(\|[^\n]*\|)', r'TABLE_ROW: \1', text)
    
    return text

def find_semantic_boundary(text: str, start: int, end: int, overlap: int) -> Optional[int]:
    """Find the best semantic boundary for chunking (optimized for speed)"""
    
    # Define boundary priorities (higher = better)
    boundary_scores = {}
    
    # Look in overlap region (limit search to last 200 chars for speed)
    search_start = max(start - 50, 0)  # Only search last 50 chars before start
    search_end = min(end, len(text))
    search_text = text[search_start:search_end]
    
    # 1. Paragraph breaks (highest priority) - fast regex
    paragraph_matches = list(re.finditer(r'\n\s*\n', search_text))
    for match in paragraph_matches:
        pos = search_start + match.end()
        if start < pos <= end:
            boundary_scores[pos] = boundary_scores.get(pos, 0) + PARAGRAPH_BREAK_WEIGHT
    
    # 2. Sentence boundaries - use fast regex instead of slow sent_tokenize
    sentence_matches = list(re.finditer(r'[.!?]\s+', search_text))
    for match in sentence_matches:
        pos = search_start + match.end()
        if start < pos <= end:
            boundary_scores[pos] = boundary_scores.get(pos, 0) + 0.6
    
    # 3. Header boundaries
    header_matches = list(re.finditer(r'HEADER:', search_text))
    for match in header_matches:
        pos = search_start + match.start()
        if start < pos <= end:
            boundary_scores[pos] = boundary_scores.get(pos, 0) + 0.7
    
    # 4. Word boundaries (lowest priority) - fast space search
    # Find last space in search region
    last_space = search_text.rfind(' ')
    if last_space > 0:
        pos = search_start + last_space + 1
        if start < pos <= end:
            boundary_scores[pos] = boundary_scores.get(pos, 0) + 0.3
    
    # Return the boundary with highest score
    if boundary_scores:
        return max(boundary_scores.items(), key=lambda x: x[1])[0]
    
    return None

def is_quality_chunk(chunk: str) -> bool:
    """Check if chunk meets quality standards for embedding"""
    
    # Minimum size check
    if len(chunk.strip()) < MIN_CHUNK_SIZE:
        return False
    
    # Maximum size check
    if len(chunk.strip()) > MAX_CHUNK_SIZE:
        return False
    
    # Content quality check
    content_chars = sum(1 for c in chunk if c.isalnum())
    if content_chars < 20:  # At least 20 alphanumeric characters
        return False
    
    # Avoid chunks with too much whitespace
    if chunk.count(' ') / len(chunk) > 0.4:
        return False
    
    # Avoid chunks that are mostly punctuation
    punct_chars = sum(1 for c in chunk if c in '.,;:!?()[]{}"\'')
    if punct_chars / len(chunk) > 0.3:
        return False
    
    return True

def post_process_chunks(chunks: List[str]) -> List[str]:
    """Post-process chunks for optimal embedding quality"""
    
    if not chunks:
        return []
    
    final_chunks = []
    
    for i, chunk in enumerate(chunks):
        # Merge very small chunks with neighbors
        if len(chunk.strip()) < MIN_CHUNK_SIZE and i < len(chunks) - 1:
            # Try to merge with next chunk
            combined = chunk + " " + chunks[i + 1]
            if len(combined.strip()) <= MAX_CHUNK_SIZE:
                chunks[i + 1] = combined
                continue
        
        # Split very large chunks
        if len(chunk.strip()) > MAX_CHUNK_SIZE:
            sub_chunks = split_large_chunk(chunk)
            final_chunks.extend(sub_chunks)
        else:
            final_chunks.append(chunk)
    
    return final_chunks

def split_large_chunk(chunk: str) -> List[str]:
    """Split chunks that are too large for embedding models"""
    
    if len(chunk) <= MAX_CHUNK_SIZE:
        return [chunk]
    
    # Try to split at sentence boundaries
    try:
        sentences = sent_tokenize(chunk)
        sub_chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk + sentence) <= MAX_CHUNK_SIZE:
                current_chunk += sentence + " "
            else:
                if current_chunk.strip():
                    sub_chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
        
        if current_chunk.strip():
            sub_chunks.append(current_chunk.strip())
        
        return sub_chunks if sub_chunks else [chunk]
        
    except Exception:
        # Fallback to simple splitting
        return [chunk[i:i+MAX_CHUNK_SIZE] for i in range(0, len(chunk), MAX_CHUNK_SIZE)]

async def generate_embeddings_for_chunks(chunks: List[str], file_id: str) -> List[Dict[str, Any]]:
    """
    Generate embeddings for text chunks using RAG service
    Creates documents with valid UUID IDs (required by Qdrant)
    """
    client = await get_http_client()
    
    documents = []
    total_chunks = len(chunks)
    logger.info(f"Preparing {total_chunks} documents with UUIDs...")
    
    for i, chunk in enumerate(chunks):
        # Generate a valid UUID for each chunk (required by Qdrant)
        chunk_id = str(uuid.uuid4())
        
        # For now, we'll use the RAG service's embedding capabilities
        # In a full implementation, you might want to call the LLM service directly
        document = {
            "id": chunk_id,  # Valid UUID format required by Qdrant
            "text": chunk,
            "metadata": {
                "file_id": file_id,
                "chunk_index": i + 1,
                "total_chunks": total_chunks,
                "chunk_size": len(chunk),
                "source": "file_upload"
            }
        }
        documents.append(document)
        
        # Log progress every 10 chunks
        if (i + 1) % 10 == 0 or (i + 1) == total_chunks:
            logger.info(f"  Prepared {i + 1}/{total_chunks} documents ({(i + 1) / total_chunks * 100:.1f}%)")
    
    logger.info(f"✅ All {len(documents)} documents prepared with UUIDs")
    return documents

# Batch size for ingestion to prevent memory issues and system hanging
INGESTION_BATCH_SIZE = 50  # Process 50 documents at a time
INGESTION_BATCH_DELAY = 0.1  # Small delay between batches to yield to event loop

async def ingest_documents_to_rag(documents: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Ingest documents into RAG service in batches to prevent system hanging.
    This prevents sending all documents at once which can overwhelm memory/network.
    """
    client = await get_http_client()
    total_docs = len(documents)
    
    # For small batches (< 50 docs), send all at once
    if total_docs <= INGESTION_BATCH_SIZE:
        return await _ingest_batch(client, documents, 0, total_docs, total_docs)
    
    # For larger batches, process in chunks
    logger.info(f"Processing {total_docs} documents in batches of {INGESTION_BATCH_SIZE}")
    total_ingested = 0
    total_failed = 0
    
    for batch_start in range(0, total_docs, INGESTION_BATCH_SIZE):
        batch_end = min(batch_start + INGESTION_BATCH_SIZE, total_docs)
        batch = documents[batch_start:batch_end]
        batch_num = (batch_start // INGESTION_BATCH_SIZE) + 1
        total_batches = (total_docs + INGESTION_BATCH_SIZE - 1) // INGESTION_BATCH_SIZE
        
        try:
            logger.info(f"📦 Batch {batch_num}/{total_batches}: Processing documents {batch_start+1}-{batch_end} of {total_docs}")
            result = await _ingest_batch(client, batch, batch_start, batch_end, total_docs)
            ingested = result.get('ingested_count', 0)
            failed = result.get('failed_count', 0)
            total_ingested += ingested
            total_failed += failed
            
            # Small delay to yield control to event loop and prevent blocking
            await asyncio.sleep(INGESTION_BATCH_DELAY)
            
        except Exception as e:
            logger.error(f"❌ Batch {batch_num} failed: {e}")
            total_failed += len(batch)
            # Continue with next batch instead of failing completely
            continue
    
    logger.info(f"✅ Batch processing complete: {total_ingested} ingested, {total_failed} failed")
    return {
        "ingested_count": total_ingested,
        "failed_count": total_failed,
        "total_documents": total_docs
    }

async def _ingest_batch(client: httpx.AsyncClient, batch: List[Dict[str, Any]], 
                       batch_start: int, batch_end: int, total_docs: int) -> Dict[str, Any]:
    """Ingest a single batch of documents"""
    try:
        response = await client.post(
            f"{RAG_SERVICE_URL}/ingest",
            json={"documents": batch},
            timeout=300.0  # 5 minute timeout per batch
        )
        
        if response.status_code == 200:
            result = response.json()
            ingested = result.get('ingested_count', 0)
            failed = result.get('failed_count', 0)
            logger.info(f"  ✅ Batch {batch_start+1}-{batch_end}: {ingested} ingested, {failed} failed")
            return result
        else:
            error_text = response.text
            logger.error(f"  ❌ Batch {batch_start+1}-{batch_end} error: {response.status_code} - {error_text}")
            
            # Check for UUID validation errors
            if "not a valid UUID" in error_text or "UUID" in error_text:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Document ID format error: {error_text}. Please ensure all document IDs are valid UUIDs."
                )
            raise HTTPException(status_code=500, detail=f"Failed to ingest batch: {error_text}")
            
    except httpx.TimeoutException:
        logger.error(f"  ⏱️ Batch {batch_start+1}-{batch_end} timeout")
        raise HTTPException(status_code=504, detail="Ingestion timeout - file too large or service too slow")
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"  ❌ Batch {batch_start+1}-{batch_end} error: {e}", exc_info=True)
        raise

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the main UI"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api")
async def api_info():
    """API information endpoint"""
    return {
        "message": "RAG Ingestion Application",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "upload": "/upload",
            "process": "/process/{file_id}",
            "files": "/files",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint - cached to reduce logging frequency"""
    cache_key = "rag_ingestion_health"
    current_time = time.time()
    
    # Return cached result if still valid
    if cache_key in _health_check_cache:
        cached_time, cached_response = _health_check_cache[cache_key]
        if current_time - cached_time < _health_check_cache_ttl:
            return cached_response
    
    client = await get_http_client()
    
    # Suppress httpx INFO logging for health checks
    import logging as httpx_logging
    original_level = httpx_logging.getLogger("httpx").level
    httpx_logging.getLogger("httpx").setLevel(logging.WARNING)
    
    try:
        # Check RAG service
        try:
            rag_response = await client.get(f"{RAG_SERVICE_URL}/health", timeout=5.0)
            rag_status = "healthy" if rag_response.status_code == 200 else "unhealthy"
        except Exception:
            rag_status = "unreachable"
        
        # Check LLM service
        try:
            llm_response = await client.get(f"{LLM_SERVICE_URL}/health", timeout=5.0)
            llm_status = "healthy" if llm_response.status_code == 200 else "unhealthy"
        except Exception:
            llm_status = "unreachable"
        
        overall_status = "healthy" if rag_status == "healthy" and llm_status == "healthy" else "degraded"
        
        response = HealthResponse(
            status=overall_status,
            rag_service=rag_status,
            llm_service=llm_status
        )
        
        # Cache the result
        _health_check_cache[cache_key] = (current_time, response)
        
        # Only log if unhealthy
        if overall_status == "degraded":
            logger.warning(f"RAG Ingestion health check shows degraded status: rag={rag_status}, llm={llm_status}")
        
        return response
    finally:
        httpx_logging.getLogger("httpx").setLevel(original_level)

@app.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """Upload a document file"""
    
    # Ensure upload directory exists with proper permissions
    UPLOAD_DIR.mkdir(exist_ok=True, mode=0o755)
    try:
        os.chmod(UPLOAD_DIR, 0o755)
    except Exception as e:
        logger.warning(f"Could not set permissions on upload directory: {e}")
    
    # Validate file type
    allowed_extensions = {'.pdf', '.docx', '.txt'}
    file_extension = Path(file.filename).suffix.lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Validate file size
    max_size_bytes = MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if file.size and file.size > max_size_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"File size exceeds maximum allowed size of {MAX_UPLOAD_SIZE_MB}MB"
        )
    
    # Generate unique file ID
    file_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{file_id}_{file.filename}"
    
    try:
        # Save uploaded file
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        # Track file for cleanup
        uploaded_files[file_id] = time.time()
        
        logger.info(f"File uploaded: {file.filename} -> {file_path} ({len(content)} bytes)")
        
        return UploadResponse(
            file_id=file_id,
            filename=file.filename,
            status="uploaded",
            message="File uploaded successfully"
        )
        
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")

@app.post("/process/{file_id}", response_model=ProcessingResponse)
async def process_file(file_id: str, background_tasks: BackgroundTasks):
    """Process uploaded file: extract text, chunk, and generate embeddings"""
    
    logger.info(f"Processing request received for file_id: {file_id}")
    
    # Find the uploaded file
    file_pattern = f"{file_id}_*"
    logger.info(f"Searching for files matching pattern: {file_pattern} in {UPLOAD_DIR}")
    
    matching_files = list(UPLOAD_DIR.glob(file_pattern))
    logger.info(f"Found {len(matching_files)} matching files")
    
    if not matching_files:
        logger.error(f"File not found for file_id: {file_id}, pattern: {file_pattern}")
        raise HTTPException(status_code=404, detail=f"File not found for file_id: {file_id}")
    
    file_path = matching_files[0]
    filename = file_path.name
    
    # Verify file exists and is readable
    if not file_path.exists():
        logger.error(f"File path does not exist: {file_path}")
        raise HTTPException(status_code=404, detail=f"File not found at path: {file_path}")
    
    if not file_path.is_file():
        logger.error(f"Path is not a file: {file_path}")
        raise HTTPException(status_code=400, detail=f"Path is not a file: {file_path}")
    
    logger.info(f"Processing file: {filename} at path: {file_path}")
    
    try:
        import time
        start_time = time.time()
        
        # Extract text from file (run in thread pool to avoid blocking)
        logger.info(f"📄 STEP 0/3: Extracting text from {filename}...")
        try:
            text = await asyncio.to_thread(extract_text_from_file, file_path)
            logger.info(f"✅ STEP 0/3: Text extraction completed, extracted {len(text)} characters")
        except Exception as e:
            logger.error(f"❌ STEP 0/3: Text extraction failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to extract text: {str(e)}")
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text content found in file")
        
        # Chunk the text using LangChain's proven RecursiveCharacterTextSplitter
        # This is industry-standard, battle-tested, and won't hang
        text_length = len(text)
        logger.info(f"Chunking text from {filename} (text length: {text_length} characters)")
        
        # Initialize LangChain text splitter with optimal settings for LLaMA-3-8B
        # Separators prioritize: paragraphs > sentences > words > characters
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,  # 1024 characters
            chunk_overlap=CHUNK_OVERLAP,  # 128 characters overlap
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]  # Priority order for splitting
        )
        
        logger.info(f"Using LangChain RecursiveCharacterTextSplitter for chunking")
        
        # Run chunking in thread pool with timeout (LangChain is fast but safety first)
        try:
            # Calculate timeout: 5 seconds per 10K chars, minimum 5s
            timeout = max(5.0, (text_length / 10000) * 5.0)
            
            chunks = await asyncio.wait_for(
                asyncio.to_thread(text_splitter.split_text, text),
                timeout=timeout
            )
            
            logger.info(f"✅ LangChain chunking completed, created {len(chunks)} chunks")
            
        except asyncio.TimeoutError:
            logger.error(f"Chunking timed out after {timeout}s - using emergency fallback")
            # Emergency fallback: simple string split (can't hang)
            if len(text) <= CHUNK_SIZE:
                chunks = [text]
            else:
                chunks = [text[i:i+CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP)]
            logger.warning(f"Emergency fallback: returning {len(chunks)} chunk(s)")
            
        except Exception as e:
            logger.error(f"Chunking failed: {e}, using emergency fallback", exc_info=True)
            # Emergency fallback: simple string split
            if len(text) <= CHUNK_SIZE:
                chunks = [text]
            else:
                chunks = [text[i:i+CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP)]
            logger.warning(f"Emergency fallback: returning {len(chunks)} chunk(s)")
        
        logger.info(f"✅ Created {len(chunks)} chunks from {filename}")
        
        if not chunks:
            raise HTTPException(status_code=400, detail="No chunks created from file text")
        
        # Generate embeddings and prepare documents
        logger.info(f"📝 STEP 1/3: Generating document structures for {len(chunks)} chunks...")
        documents = await generate_embeddings_for_chunks(chunks, file_id)
        logger.info(f"✅ STEP 1/3: Generated {len(documents)} document structures")
        
        # Ingest into RAG service
        logger.info(f"🚀 STEP 2/3: Ingesting {len(documents)} documents to RAG service (this may take 1-5 minutes for large files)...")
        ingestion_start = time.time()
        try:
            ingestion_result = await ingest_documents_to_rag(documents)
            ingestion_time = time.time() - ingestion_start
            logger.info(f"✅ STEP 2/3: Ingestion completed in {ingestion_time:.2f}s - Result: {ingestion_result}")
        except Exception as e:
            ingestion_time = time.time() - ingestion_start
            logger.error(f"❌ STEP 2/3: Ingestion failed after {ingestion_time:.2f}s - Error: {e}")
            raise
        
        processing_time = time.time() - start_time
        
        logger.info(f"Processing completed for {filename}: {len(chunks)} chunks in {processing_time:.2f}s")
        
        return ProcessingResponse(
            file_id=file_id,
            chunks_created=len(chunks),
            embeddings_generated=len(documents),
            processing_time=processing_time,
            status="completed"
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is (they're already properly formatted)
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error processing file {filename}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to process file: {str(e)}. Check logs for details."
        )

@app.get("/files")
async def list_uploaded_files():
    """List all uploaded files"""
    files = []
    for file_path in UPLOAD_DIR.glob("*"):
        if file_path.is_file():
            files.append({
                "filename": file_path.name,
                "size": file_path.stat().st_size,
                "uploaded_at": file_path.stat().st_mtime
            })
    
    return {"files": files}

@app.delete("/files/{file_id}")
async def delete_file(file_id: str):
    """Delete uploaded file"""
    file_pattern = f"{file_id}_*"
    matching_files = list(UPLOAD_DIR.glob(file_pattern))
    
    if not matching_files:
        raise HTTPException(status_code=404, detail="File not found")
    
    file_path = matching_files[0]
    
    try:
        file_path.unlink()
        # Remove from tracking
        if file_id in uploaded_files:
            del uploaded_files[file_id]
        logger.info(f"File deleted: {file_path}")
        return {"message": "File deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting file {file_path}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete file: {str(e)}")

@app.post("/cleanup")
async def manual_cleanup():
    """Manually trigger file cleanup"""
    try:
        cleanup_old_files()
        return {"message": "Cleanup completed successfully"}
    except Exception as e:
        logger.error(f"Manual cleanup failed: {e}")
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")

@app.get("/storage-info")
async def get_storage_info():
    """Get storage information"""
    try:
        total_files = 0
        total_size = 0
        file_types = {}
        
        for file_path in UPLOAD_DIR.glob("*"):
            if file_path.is_file():
                total_files += 1
                file_size = file_path.stat().st_size
                total_size += file_size
                
                file_ext = file_path.suffix.lower()
                file_types[file_ext] = file_types.get(file_ext, 0) + 1
        
        return {
            "total_files": total_files,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "file_types": file_types,
            "retention_hours": FILE_RETENTION_HOURS,
            "cleanup_interval_hours": CLEANUP_INTERVAL_HOURS,
            "max_upload_size_mb": MAX_UPLOAD_SIZE_MB
        }
    except Exception as e:
        logger.error(f"Error getting storage info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get storage info: {str(e)}")

if __name__ == "__main__":
    # Start periodic cleanup task
    import asyncio
    
    async def start_app():
        # Start cleanup task
        cleanup_task = asyncio.create_task(periodic_cleanup())
        
        # Start the server
        config = uvicorn.Config(app, host="0.0.0.0", port=8005)
        server = uvicorn.Server(config)
        await server.serve()
        
        # Cancel cleanup task when server stops
        cleanup_task.cancel()
    
    # Run the application
    asyncio.run(start_app())
