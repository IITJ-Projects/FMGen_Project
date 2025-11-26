# Conversational AI Agent - Service Architecture

## 📁 Folder Structure

```
zevo-ai/
├── docker-compose.yml              # Main orchestration file
├── README.md                       # Deployment guide
├── SERVICES_STRUCTURE.md           # This file
│
├── asr_service/                    # Speech Recognition Service
│   ├── main.py                     # FastAPI application
│   ├── requirements.txt            # Python dependencies
│   └── Dockerfile                  # Container definition
│
├── tts_service/                    # Text-to-Speech Service
│   ├── main.py                     # FastAPI application
│   ├── requirements.txt            # Python dependencies
│   └── Dockerfile                  # Container definition
│
├── orchestration_service/          # Pipeline Orchestrator
│   ├── main.py                     # FastAPI application
│   ├── requirements.txt            # Python dependencies
│   └── Dockerfile                  # Container definition
│
├── llm_service/                    # Language Model Service
│   ├── main.py                     # vLLM FastAPI application
│   ├── requirements.txt            # Python dependencies
│   └── Dockerfile                  # Container definition
│
├── rag_service/                    # RAG Service
│   ├── main.py                     # FastAPI application
│   ├── requirements.txt            # Python dependencies
│   └── Dockerfile                  # Container definition
│
├── logs/                           # Shared log directory
├── config/                         # Configuration files
└── data/                           # Data storage
    ├── models/                     # Model caches
    ├── embeddings/                 # Embedding models
    └── reranker/                   # Reranker models
```

## Service Details

### 1. ASR Service (Port 8001)

**Purpose**: Convert speech to text using faster-whisper-medium

**Endpoints**:

- `POST /transcribe` - Transcribe audio file
- `POST /transcribe_text` - Mock endpoint for testing
- `GET /health` - Health check

**Features**:

- Audio file upload (multipart)
- Multiple audio formats support
- Language detection
- Confidence scoring
- Model caching
- Error handling

### 2. TTS Service (Port 8003)

**Purpose**: Convert text to speech using MeloTTS

**Endpoints**:

- `POST /speak` - Generate speech from text
- `POST /speak_stream` - Stream audio chunks
- `POST /speak_status` - Validate text
- `GET /health` - Health check

**Features**:

- Text-to-speech synthesis
- Streaming audio support
- Multiple voice options
- Speed control
- Fallback to gTTS
- Audio format conversion

### 3. Orchestration Service (Port 8000)

**Purpose**: Coordinate the entire conversational AI pipeline

**Endpoints**:

- `POST /chat` - Main chat endpoint (audio in → audio out)
- `POST /conversation/start` - Start new session
- `GET /conversation/{session_id}/history` - Get chat history
- `GET /health` - Health check with service status

**Pipeline Flow**:

```
Audio Input → ASR → RAG → LLM → TTS → Audio Output
```

**Features**:

- Complete pipeline orchestration
- Session management
- Service health monitoring
- Conversation history
- Error handling and fallbacks
- Background task processing

### 4. LLM Service (Port 8002)

**Purpose**: High-throughput language model inference using vLLM

**Endpoints**:

- `POST /generate` - Generate text response
- `POST /generate_stream` - Stream generated tokens
- `POST /chat` - OpenAI-compatible chat completion
- `GET /models` - List available models
- `GET /health` - Health check

**Features**:

- vLLM high-throughput inference
- LLaMA-3-8B-Instruct model support
- AWQ quantization for efficiency
- Streaming token generation
- OpenAI-compatible API
- Configurable sampling parameters
- GPU memory optimization

### 5. RAG Service (Port 8004)

**Purpose**: Retrieval-Augmented Generation with vector search

**Endpoints**:

- `POST /retrieve` - Retrieve relevant documents
- `POST /ingest` - Ingest documents into vector DB
- `DELETE /documents/{id}` - Delete document
- `GET /documents/count` - Get document count
- `POST /embed` - Generate embeddings
- `GET /health` - Health check

**Features**:

- Qdrant vector database integration
- Sentence transformers embeddings
- BGE reranker for improved relevance
- Document ingestion and management
- Configurable search parameters
- Score threshold filtering
- Metadata support

## Deployment

### 1. Build Docker Images

```bash
# Build all services
docker build -t zevo-ai/asr-service:latest -f asr_service/Dockerfile .
docker build -t zevo-ai/tts-service:latest -f tts_service/Dockerfile .
docker build -t zevo-ai/orchestration-service:latest -f orchestration_service/Dockerfile .
docker build -t zevo-ai/llm-service:latest -f llm_service/Dockerfile .
docker build -t zevo-ai/rag-service:latest -f rag_service/Dockerfile .
```

### 2. Start Services

```bash
# Start all services
docker-compose up -d

# Monitor logs
docker-compose logs -f

# Check health
curl http://localhost:8000/health
```

### 3. Test Endpoints

```bash
# Test ASR
curl -X POST http://localhost:8001/transcribe \
  -F "file=@audio.wav"

# Test TTS
curl -X POST http://localhost:8003/speak \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world"}'

# Test LLM
curl -X POST http://localhost:8002/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is artificial intelligence?", "max_tokens": 100}'

# Test RAG
curl -X POST http://localhost:8004/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "machine learning", "top_k": 5}'

# Test Chat Pipeline
curl -X POST http://localhost:8000/chat \
  -F "file=@audio.wav" \
  -F "session_id=test123"
```

## 🔗 Service Communication

### Internal Network

All services communicate via Docker network `zevo-network`:

- ASR Service: `http://asr-service:8001`
- LLM Service: `http://llm-service:8002`
- TTS Service: `http://tts-service:8003`
- RAG Service: `http://rag-service:8004`
- Orchestration: `http://orchestration-service:8000`
- Qdrant DB: `http://qdrant-db:6333`

### Health Checks

Each service provides `/health` endpoint:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu"
}
```

## Monitoring

### Logs

- All services log to `/app/logs` (mounted to `./logs`)
- Structured logging with timestamps
- Error tracking and debugging

### Health Monitoring

- Service health checks every 30-60 seconds
- Automatic restart on failure
- Health status aggregation

## Security

### Production Considerations

- Add authentication/authorization
- Implement rate limiting
- Use HTTPS/TLS
- Add input validation
- Implement proper error handling

### Environment Variables

- Model configurations
- Service URLs
- Log levels
- Performance settings

## Next Steps

1. **Add Dockerfiles** - Container definitions for each service
2. **Implement Authentication** - User management and security
3. **Add Monitoring** - Metrics, alerts, and dashboards
4. **Optimize Performance** - Caching, load balancing, scaling
5. **Add Model Management** - Model versioning and updates
6. **Implement Caching** - Redis for session and response caching

## Notes

- All services are production-ready with proper error handling
- Async/await patterns used throughout for performance
- Comprehensive logging and health checks
- Modular design for easy maintenance and scaling
- Docker Compose integration for seamless deployment
