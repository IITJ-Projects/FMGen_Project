#!/usr/bin/env python3
"""
RAG Service - Retrieval-Augmented Generation Microservice
Uses Qdrant vector database with sentence transformers and reranker
"""

import os
import logging
import asyncio
import uuid
import time
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel, Field, validator
import httpx
from sentence_transformers import SentenceTransformer
import torch
import numpy as np

# Reranker imports
try:
    from sentence_transformers import CrossEncoder
    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False

# Language detection imports
try:
    from langdetect import detect, detect_langs, DetectorFactory
    # Set seed for consistent results
    DetectorFactory.seed = 0
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Log warning about missing langdetect after logger is defined
if not LANGDETECT_AVAILABLE:
    logger.warning("langdetect not available. Language detection will be disabled.")

# Global model instances
bge_embedding_model = None  # BGE-Large-EN-v1.5 for English
multilingual_embedding_model = None  # multilingual-E5-Large for multilingual
reranker_model = None
qdrant_client = None

# English language codes
ENGLISH_LANGS = {'en'}

class Document(BaseModel):
    id: str = Field(..., description="Unique document identifier (must be a valid UUID)")
    text: str = Field(..., description="Document text content")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    
    @validator('id')
    def validate_uuid(cls, v):
        """Validate that id is a valid UUID format"""
        try:
            uuid.UUID(v)
            return v
        except (ValueError, TypeError):
            raise ValueError(f"Document ID '{v}' is not a valid UUID. Qdrant requires UUID format (e.g., '550e8400-e29b-41d4-a716-446655440000')")

class RetrieveRequest(BaseModel):
    query: str = Field(..., description="Search query")
    top_k: int = Field(10, ge=1, le=100, description="Number of documents to retrieve")
    score_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum similarity score")
    use_reranker: bool = Field(True, description="Enable reranking")
    language: Optional[str] = Field("en", description="Language hint ('en' for English, 'multilingual' for others). Defaults to 'en'")

class RetrieveResponse(BaseModel):
    documents: List[Dict[str, Any]]
    query: str
    total_found: int
    processing_time: float

class IngestRequest(BaseModel):
    documents: List[Document] = Field(..., description="Documents to ingest")

class IngestResponse(BaseModel):
    ingested_count: int
    failed_count: int
    processing_time: float

class EmbedRequest(BaseModel):
    texts: List[str] = Field(..., description="List of texts to embed")
    language: Optional[str] = Field("en", description="Language hint ('en' for English, 'multilingual' for others). Defaults to 'en'")

class EmbedResponse(BaseModel):
    embeddings: List[List[float]]

class HealthResponse(BaseModel):
    status: str
    bge_model_loaded: bool
    multilingual_model_loaded: bool
    reranker_model_loaded: bool
    qdrant_connected: bool
    bge_model_name: str
    multilingual_model_name: str
    reranker_model_name: str

class QdrantClient:
    """Qdrant client wrapper"""
    
    def __init__(self, host: str, port: int, collection_name: str = "documents"):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.base_url = f"http://{host}:{port}"
        self.http_client = None
    
    async def get_client(self):
        """Get HTTP client"""
        if self.http_client is None:
            self.http_client = httpx.AsyncClient(timeout=30.0)
        return self.http_client
    
    async def health_check(self) -> bool:
        """Check if Qdrant is healthy with fallback paths"""
        try:
            client = await self.get_client()
            # Try /healthz
            for path in ["/healthz", "/readyz"]:
                try:
                    response = await client.get(f"{self.base_url}{path}")
                    if response.status_code == 200:
                        return True
                except Exception:
                    pass
            # Fallback: attempt a lightweight collections call
            response = await client.get(f"{self.base_url}/collections")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Qdrant health check failed: {e}")
            return False
    
    async def create_collection(self) -> bool:
        """Create collection if it doesn't exist or has wrong dimensions"""
        try:
            client = await self.get_client()
            required_dimension = 1024  # BGE-Large-EN-v1.5 and multilingual-E5-Large dimensions
            
            # Check if collection exists
            response = await client.get(f"{self.base_url}/collections/{self.collection_name}")
            if response.status_code == 200:
                collection_info = response.json()
                result = collection_info.get("result", collection_info)
                current_config = result.get("config", {})
                params = current_config.get("params", {})
                
                # Handle both single vector config and named vectors
                vectors_config = params.get("vectors", {})
                if isinstance(vectors_config, dict) and "size" in vectors_config:
                    vector_size = vectors_config.get("size")
                elif isinstance(vectors_config, dict):
                    # Named vectors - check first vector config
                    first_vector = next(iter(vectors_config.values()), {})
                    vector_size = first_vector.get("size") if isinstance(first_vector, dict) else None
                else:
                    vector_size = None
                
                if vector_size == required_dimension:
                    logger.info(f"Collection {self.collection_name} already exists with correct dimension ({required_dimension})")
                    return True
                elif vector_size:
                    logger.warning(
                        f"Collection {self.collection_name} exists with wrong dimension "
                        f"({vector_size} instead of {required_dimension}). Deleting and recreating..."
                    )
                    # Delete old collection with wrong dimensions
                    delete_response = await client.delete(
                        f"{self.base_url}/collections/{self.collection_name}"
                    )
                    if delete_response.status_code == 200:
                        logger.info(f"Deleted old collection {self.collection_name} with wrong dimensions")
                    else:
                        logger.error(f"Failed to delete old collection: {delete_response.text}")
                        return False
                else:
                    logger.warning(f"Could not determine vector dimension, recreating collection {self.collection_name}")
                    # Delete and recreate to be safe
                    delete_response = await client.delete(
                        f"{self.base_url}/collections/{self.collection_name}"
                    )
                    if delete_response.status_code != 200:
                        logger.error(f"Failed to delete collection: {delete_response.text}")
                        return False
            
            # Create collection with 1024 dimensions for BGE/multilingual-E5
            collection_config = {
                "vectors": {
                    "size": required_dimension,
                    "distance": "Cosine"
                }
            }
            
            response = await client.put(
                f"{self.base_url}/collections/{self.collection_name}",
                json=collection_config
            )
            
            if response.status_code == 200:
                logger.info(f"Created collection {self.collection_name} with dimension {required_dimension}")
                return True
            else:
                logger.error(f"Failed to create collection: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            return False
    
    async def upsert_documents(self, documents: List[Dict[str, Any]], embeddings: List[List[float]]) -> bool:
        """Upsert documents with embeddings"""
        try:
            client = await self.get_client()
            
            # Prepare points for upsert
            points = []
            for doc, embedding in zip(documents, embeddings):
                point = {
                    "id": doc["id"],  # Already validated as UUID
                    "vector": embedding,
                    "payload": {
                        "text": doc["text"],
                        "metadata": doc.get("metadata", {})
                    }
                }
                points.append(point)
            
            # Upsert points
            response = await client.put(
                f"{self.base_url}/collections/{self.collection_name}/points",
                json={"points": points}
            )
            
            if response.status_code == 200:
                logger.info(f"Upserted {len(points)} documents")
                return True
            else:
                logger.error(f"Failed to upsert documents: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error upserting documents: {e}")
            return False
    
    async def search_documents(self, query_embedding: List[float], top_k: int, score_threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """Search documents by embedding"""
        try:
            client = await self.get_client()
            
            # Prepare search request
            search_params = {
                "vector": query_embedding,
                "limit": top_k,
                "with_payload": True
            }
            
            if score_threshold is not None:
                search_params["score_threshold"] = score_threshold
            
            logger.info(f"Searching in collection '{self.collection_name}' with top_k={top_k}")
            
            # Search
            response = await client.post(
                f"{self.base_url}/collections/{self.collection_name}/points/search",
                json=search_params
            )
            
            if response.status_code == 200:
                result = response.json()
                documents = []
                
                search_results = result.get("result", [])
                logger.info(f"Found {len(search_results)} documents in search results")
                
                for point in search_results:
                    doc = {
                        "id": point["id"],  # Already a UUID
                        "text": point["payload"]["text"],
                        "metadata": point["payload"].get("metadata", {}),
                        "score": point["score"]
                    }
                    documents.append(doc)
                
                return documents
            else:
                logger.error(f"Search failed: {response.status_code} - {response.text}")
                return []
                
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []

def detect_language(text: str) -> str:
    """
    Detect language of text
    Returns 'en' for English (default), 'multilingual' for other languages
    English is the default for all cases: detection unavailable, text too short, or detection failures
    """
    # Default to English in all edge cases
    if not LANGDETECT_AVAILABLE:
        return 'en'
    
    if not text or len(text.strip()) < 10:
        # Default to English if text too short
        return 'en'
    
    try:
        detected_lang = detect(text)
        if detected_lang in ENGLISH_LANGS:
            return 'en'
        else:
            return 'multilingual'
    except Exception as e:
        # Default to English on any detection failure
        logger.warning(f"Language detection failed: {e}, defaulting to English")
        return 'en'

def get_embedding_model(language: str):
    """
    Get appropriate embedding model based on language
    Returns BGE for English, multilingual-E5 for other languages
    """
    if language == 'en':
        return bge_embedding_model
    else:
        return multilingual_embedding_model

def load_models():
    """Load dual embedding models and reranker"""
    global bge_embedding_model, multilingual_embedding_model, reranker_model
    
    try:
        # Load BGE model for English
        bge_model_name = os.getenv("BGE_MODEL_NAME", "BAAI/bge-large-en-v1.5")
        logger.info(f"Loading BGE embedding model (English): {bge_model_name}")
        
        bge_embedding_model = SentenceTransformer(
            bge_model_name,
            cache_folder="/app/embedding_models"
        )
        logger.info("BGE model loaded successfully")
        
        # Load multilingual-E5 model for non-English
        multilingual_model_name = os.getenv("MULTILINGUAL_MODEL_NAME", "intfloat/multilingual-e5-large")
        logger.info(f"Loading multilingual-E5 embedding model: {multilingual_model_name}")
        
        multilingual_embedding_model = SentenceTransformer(
            multilingual_model_name,
            cache_folder="/app/embedding_models"
        )
        logger.info("Multilingual-E5 model loaded successfully")
        
        # Load reranker model
        if RERANKER_AVAILABLE:
            reranker_model_name = os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-large")
            logger.info(f"Loading reranker model: {reranker_model_name}")
            # CrossEncoder in some versions does not accept cache_folder; rely on HF cache envs
            reranker_model = CrossEncoder(reranker_model_name)
            logger.info("Reranker model loaded successfully")
        
        logger.info("All models loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        return False

def initialize_qdrant():
    """Initialize Qdrant client"""
    global qdrant_client
    
    try:
        qdrant_host = os.getenv("QDRANT_HOST", "qdrant-db")
        qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
        collection_name = os.getenv("QDRANT_COLLECTION", "documents")
        
        qdrant_client = QdrantClient(qdrant_host, qdrant_port, collection_name)
        logger.info(f"Initialized Qdrant client: {qdrant_host}:{qdrant_port}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize Qdrant client: {e}")
        return False

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting RAG Service...")
    
    # Load models
    models_loaded = load_models()
    if not models_loaded:
        logger.error("Failed to load models during startup")
    
    # Initialize Qdrant
    qdrant_initialized = initialize_qdrant()
    if qdrant_initialized and qdrant_client:
        # Create collection
        await qdrant_client.create_collection()
    
    yield
    
    # Shutdown
    logger.info("Shutting down RAG Service...")
    if qdrant_client and qdrant_client.http_client:
        await qdrant_client.http_client.aclose()

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

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="RAG Service",
    description="Retrieval-Augmented Generation Service with Qdrant",
    version="1.0.0",
    lifespan=lifespan
)

# Add middleware to suppress health check logging
app.add_middleware(SuppressHealthLogMiddleware)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint - cached to reduce logging frequency"""
    cache_key = "rag_health"
    current_time = time.time()
    
    # Return cached result if still valid
    if cache_key in _health_check_cache:
        cached_time, cached_response = _health_check_cache[cache_key]
        if current_time - cached_time < _health_check_cache_ttl:
            return cached_response
    
    # Suppress httpx INFO logging for health checks
    import logging as httpx_logging
    original_level = httpx_logging.getLogger("httpx").level
    httpx_logging.getLogger("httpx").setLevel(logging.WARNING)
    
    try:
        qdrant_healthy = False
        if qdrant_client:
            qdrant_healthy = await qdrant_client.health_check()
        
        status = "healthy" if (bge_embedding_model and multilingual_embedding_model and qdrant_healthy) else "degraded"
        
        response = HealthResponse(
            status=status,
            bge_model_loaded=bge_embedding_model is not None,
            multilingual_model_loaded=multilingual_embedding_model is not None,
            reranker_model_loaded=reranker_model is not None,
            qdrant_connected=qdrant_healthy,
            bge_model_name=os.getenv("BGE_MODEL_NAME", "BAAI/bge-large-en-v1.5"),
            multilingual_model_name=os.getenv("MULTILINGUAL_MODEL_NAME", "intfloat/multilingual-e5-large"),
            reranker_model_name=os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-large")
        )
        
        # Cache the result
        _health_check_cache[cache_key] = (current_time, response)
        
        # Only log if unhealthy
        if status == "degraded":
            logger.warning(f"RAG health check shows degraded status: bge={bge_embedding_model is not None}, multilingual={multilingual_embedding_model is not None}, qdrant={qdrant_healthy}")
        
        return response
    finally:
        httpx_logging.getLogger("httpx").setLevel(original_level)

@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve_documents(request: RetrieveRequest):
    """
    Retrieve relevant documents for a query using dual embeddings
    Uses provided language hint if available, otherwise auto-detects (defaults to English)
    """
    if bge_embedding_model is None or multilingual_embedding_model is None:
        raise HTTPException(status_code=503, detail="Embedding models not loaded")
    
    if qdrant_client is None:
        raise HTTPException(status_code=503, detail="Qdrant client not initialized")
    
    try:
        import time
        start_time = time.time()
        
        # Use provided language hint, or auto-detect (defaults to 'en')
        if request.language and request.language in ['en', 'multilingual']:
            query_language = request.language
            language_source = "provided hint"
        else:
            # Auto-detect language (defaults to 'en' on failure)
            query_language = detect_language(request.query)
            language_source = "auto-detected"
        
        embedding_model = get_embedding_model(query_language)
        
        logger.info(f"Query language {language_source}: {query_language}, using {'BGE' if query_language == 'en' else 'multilingual-E5'} model")
        
        # Generate query embedding using appropriate model
        query_embedding = embedding_model.encode(request.query).tolist()
        logger.info(f"Generated query embedding (dimension: {len(query_embedding)})")
        
        # Search documents
        documents = await qdrant_client.search_documents(
            query_embedding,
            request.top_k,
            request.score_threshold
        )
        logger.info(f"Retrieved {len(documents)} documents from search")
        
        # Apply reranking if enabled and model available
        if request.use_reranker and reranker_model and documents:
            logger.info("Applying reranking...")
            
            # Prepare pairs for reranking
            pairs = [(request.query, doc["text"]) for doc in documents]
            
            # Get reranker scores
            scores = reranker_model.predict(pairs)
            
            # Update documents with reranker scores
            for doc, score in zip(documents, scores):
                doc["reranker_score"] = float(score)
            
            # Sort by reranker score
            documents.sort(key=lambda x: x["reranker_score"], reverse=True)
        
        processing_time = time.time() - start_time
        
        logger.info(f"Retrieved {len(documents)} documents in {processing_time:.2f}s")
        
        return RetrieveResponse(
            documents=documents,
            query=request.query,
            total_found=len(documents),
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        raise HTTPException(status_code=500, detail=f"Document retrieval failed: {str(e)}")

@app.post("/ingest", response_model=IngestResponse)
async def ingest_documents(request: IngestRequest):
    """
    Ingest documents into the vector database using dual embeddings
    Automatically detects language for each document and uses appropriate embedding model
    """
    if bge_embedding_model is None or multilingual_embedding_model is None:
        raise HTTPException(status_code=503, detail="Embedding models not loaded")
    
    if qdrant_client is None:
        raise HTTPException(status_code=503, detail="Qdrant client not initialized")
    
    try:
        import time
        start_time = time.time()
        
        # Prepare documents with language detection
        docs = []
        embeddings = []
        bge_count = 0
        multilingual_count = 0
        
        total_docs = len(request.documents)
        for idx, doc in enumerate(request.documents):
            # Detect language for each document
            doc_language = detect_language(doc.text)
            
            # Get appropriate embedding model
            embedding_model = get_embedding_model(doc_language)
            
            # Generate embedding
            embedding = embedding_model.encode(doc.text).tolist()
            
            # Prepare document with language metadata
            doc_dict = {
                "id": doc.id,
                "text": doc.text,
                "metadata": {
                    **(doc.metadata or {}),
                    "language": doc_language,
                    "embedding_model": "BGE" if doc_language == 'en' else "multilingual-E5"
                }
            }
            
            docs.append(doc_dict)
            embeddings.append(embedding)
            
            if doc_language == 'en':
                bge_count += 1
            else:
                multilingual_count += 1
            
            # Log progress every 10 documents
            if (idx + 1) % 10 == 0 or (idx + 1) == total_docs:
                logger.info(f"Processed {idx + 1}/{total_docs} documents ({((idx + 1) / total_docs * 100):.1f}%)")
        
        logger.info(f"Ingesting {len(docs)} documents: {bge_count} English (BGE), {multilingual_count} multilingual (E5)")
        
        # Upsert documents
        success = await qdrant_client.upsert_documents(docs, embeddings)
        
        processing_time = time.time() - start_time
        
        if success:
            logger.info(f"Ingested {len(docs)} documents in {processing_time:.2f}s")
            
            return IngestResponse(
                ingested_count=len(docs),
                failed_count=0,
                processing_time=processing_time
            )
        else:
            return IngestResponse(
                ingested_count=0,
                failed_count=len(docs),
                processing_time=processing_time
            )
        
    except Exception as e:
        logger.error(f"Ingestion error: {e}")
        raise HTTPException(status_code=500, detail=f"Document ingestion failed: {str(e)}")

@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """
    Delete a document from the vector database
    document_id must be a valid UUID
    """
    if qdrant_client is None:
        raise HTTPException(status_code=503, detail="Qdrant client not initialized")
    
    # Validate UUID format
    try:
        uuid.UUID(document_id)
    except (ValueError, TypeError):
        raise HTTPException(
            status_code=400, 
            detail=f"Document ID '{document_id}' is not a valid UUID. Qdrant requires UUID format."
        )
    
    try:
        client = await qdrant_client.get_client()
        
        response = await client.delete(
            f"{qdrant_client.base_url}/collections/{qdrant_client.collection_name}/points",
            json={"points": [document_id]}
        )
        
        if response.status_code == 200:
            logger.info(f"Deleted document: {document_id}")
            return {"message": f"Document {document_id} deleted successfully"}
        else:
            logger.error(f"Failed to delete document: {response.text}")
            raise HTTPException(status_code=500, detail="Failed to delete document")
        
    except Exception as e:
        logger.error(f"Delete error: {e}")
        raise HTTPException(status_code=500, detail=f"Document deletion failed: {str(e)}")

@app.get("/documents/count")
async def get_document_count():
    """
    Get total number of documents in the collection
    """
    if qdrant_client is None:
        raise HTTPException(status_code=503, detail="Qdrant client not initialized")
    
    try:
        client = await qdrant_client.get_client()
        
        collection_url = f"{qdrant_client.base_url}/collections/{qdrant_client.collection_name}"
        logger.info(f"Getting document count from: {collection_url}")
        
        response = await client.get(collection_url)
        
        logger.info(f"Qdrant response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            count = result.get("points_count", 0)
            logger.info(f"Collection '{qdrant_client.collection_name}' has {count} documents")
            return {"total_documents": count}
        else:
            logger.error(f"Failed to get document count: {response.status_code} - {response.text}")
            raise HTTPException(status_code=500, detail=f"Failed to get document count: {response.text}")
        
    except Exception as e:
        logger.error(f"Count error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get document count: {str(e)}")

@app.post("/embed", response_model=EmbedResponse)
async def embed_text(request: EmbedRequest):
    """
    Generate embeddings for given texts using dual embeddings
    Uses provided language hint (defaults to 'en') for all texts, or detects language per text if None
    """
    if bge_embedding_model is None or multilingual_embedding_model is None:
        raise HTTPException(status_code=503, detail="Embedding models not loaded")
    
    try:
        # Use provided language hint (defaults to 'en'), or detect per text
        if request.language and request.language in ['en', 'multilingual']:
            # Use specified language model for all texts
            embedding_model = get_embedding_model(request.language)
            embeddings = embedding_model.encode(request.texts).tolist()
        else:
            # Detect language per text (defaults to 'en' on failure)
            embeddings = []
            for text in request.texts:
                text_language = detect_language(text)
                embedding_model = get_embedding_model(text_language)
                embedding = embedding_model.encode(text).tolist()
                embeddings.append(embedding)
        
        return EmbedResponse(embeddings=embeddings)
        
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8004,
        reload=False,
        log_level="info"
    )
