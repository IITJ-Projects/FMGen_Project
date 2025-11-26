# Zevo AI - End-to-End Workflow Diagram

## Complete End-to-End Workflow for Text Mode

**Zevo AI - End-to-End Workflow**: Text Input → AI Processing → Audio Output

## Step 1: User Interface Layer

```mermaid
graph LR
    subgraph Frontend["<b>Frontend App</b> <i>Port: 8080</i>"]
        direction LR
        UserInput["<b>User Input</b><br/><i>Text Message</i><br/><i>Send Button</i><br/><i>Character Count</i><br/><i>Mode Toggle</i>"]
        Connection["<b>Connection Mgmt</b><br/><i>WebSocket</i><br/><i>WebRTC</i><br/><i>Health Check</i><br/><i>Status Update</i>"]
        AudioOutput["<b>Audio Output</b><br/><i>HTML5 Audio</i><br/><i>Chunk Player</i><br/><i>Volume Control</i><br/><i>Playback Queue</i>"]
    end

    Tech["<i>Technologies: HTML5, JavaScript, WebSocket API, WebRTC API</i>"]

    style Frontend fill:#e1f5ff,stroke:#01579b,stroke-width:2px,color:#000
    style UserInput fill:#e3f2fd,stroke:#0d47a1,stroke-width:1px,color:#000
    style Connection fill:#e3f2fd,stroke:#0d47a1,stroke-width:1px,color:#000
    style AudioOutput fill:#e3f2fd,stroke:#0d47a1,stroke-width:1px,color:#000
```

## Step 2: Communication Layer

```mermaid
graph LR
    subgraph Protocols["<b>Protocol Routing</b>"]
        direction LR
        TextMode["<b>Text Mode</b><br/><i>WebSocket</i><br/><i>wss://agent.zevo360.in</i><br/><i>/ws/chat/{session_id}</i>"]
        VoiceMode["<b>Voice Mode</b><br/><i>WebRTC Data Channel</i><br/><i>+ WebSocket fallback</i>"]
        HealthCheck["<b>Health Check</b><br/><i>HTTP REST</i><br/><i>/health</i>"]
    end

    SessionID["<i>Session ID: session_{timestamp}_{random_string}</i><br/><i>Example: session_1758281032749_sle20ndep</i>"]

    style Protocols fill:#fff3e0,stroke:#e65100,stroke-width:2px,color:#000
    style TextMode fill:#e3f2fd,stroke:#0d47a1,stroke-width:1px,color:#000
    style VoiceMode fill:#e3f2fd,stroke:#0d47a1,stroke-width:1px,color:#000
    style HealthCheck fill:#e3f2fd,stroke:#0d47a1,stroke-width:1px,color:#000
```

## Step 3: Orchestration Service

```mermaid
graph LR
    subgraph Orchestration["<b>Orchestration Service</b> <i>Port: 8000</i>"]
        direction LR
        SessionMgmt["<b>Session Mgmt</b><br/><i>Store History</i><br/><i>Track State</i><br/><i>Error Handle</i>"]
        ContextPrep["<b>Context Prep</b><br/><i>System Prompt</i><br/><i>Recent Chat</i><br/><i>Context Merge</i>"]
        LatencyTrack["<b>Latency Track</b><br/><i>Step Timing</i><br/><i>Performance</i><br/><i>Optimization</i>"]
    end

    Input["<i>Input: Text Message + Session ID + History</i>"]
    Output["<i>Output: Audio Chunks + Text Response + Status</i>"]
    Tech["<i>Technologies: FastAPI, Python, AsyncIO, httpx, WebSocket</i>"]

    Input --> Orchestration
    Orchestration --> Output

    style Orchestration fill:#fff3e0,stroke:#e65100,stroke-width:3px,color:#000
    style SessionMgmt fill:#e3f2fd,stroke:#0d47a1,stroke-width:1px,color:#000
    style ContextPrep fill:#e3f2fd,stroke:#0d47a1,stroke-width:1px,color:#000
    style LatencyTrack fill:#e3f2fd,stroke:#0d47a1,stroke-width:1px,color:#000
```

## Step 4: RAG Service (Context Retrieval)

```mermaid
graph LR
    subgraph RAG["<b>RAG Service</b> <i>Port: 8004</i>"]
        direction LR
        Embedding["<b>Embedding Gen</b><br/><i>BGE-Large</i><br/><i>Multilingual</i><br/><i>1024-dim</i>"]
        VectorSearch["<b>Vector Search</b><br/><i>Qdrant Query</i><br/><i>Similarity</i><br/><i>Top-K Results</i>"]
        Reranking["<b>Reranking</b><br/><i>BGE-Reranker</i><br/><i>Score Fusion</i><br/><i>Context Rank</i>"]
    end

    Input["<i>Input: User Query + Session Context</i>"]
    Output["<i>Output: Relevant Context + Confidence Scores</i>"]
    Models["<i>Models: BGE-Large-EN-v1.5, multilingual-E5-Large, BGE-Reranker-Large</i>"]
    Tech["<i>Technologies: sentence-transformers, Qdrant, Python, FastAPI</i>"]

    Input --> Embedding
    Embedding --> VectorSearch
    VectorSearch --> Reranking
    Reranking --> Output

    style RAG fill:#fff9c4,stroke:#f57f17,stroke-width:2px,color:#000
    style Embedding fill:#e3f2fd,stroke:#0d47a1,stroke-width:1px,color:#000
    style VectorSearch fill:#e3f2fd,stroke:#0d47a1,stroke-width:1px,color:#000
    style Reranking fill:#e3f2fd,stroke:#0d47a1,stroke-width:1px,color:#000
```

## Step 5: LLM Service (Text Generation)

```mermaid
graph LR
    subgraph LLM["<b>LLM Service</b> <i>Port: 8002</i>"]
        direction LR
        Model["<b>Model Loading</b><br/><i>LLaMA-3-8B</i><br/><i>AWQ Quantized</i><br/><i>4K Context</i>"]
        Inference["<b>Inference Engine</b><br/><i>vLLM Engine</i><br/><i>GPU Accelerated</i><br/><i>Batch Processing</i>"]
        Streaming["<b>Token Streaming</b><br/><i>Real-time</i><br/><i>Chunked Output</i><br/><i>Stop Conditions</i>"]
    end

    Input["<i>Input: Prompt + Context + History</i>"]
    Output["<i>Output: Streaming Tokens + Full Response</i>"]
    Details["<i>Model: meta-llama/Meta-Llama-3-8B-Instruct</i><br/><i>AWQ 4-bit, 4096 tokens, Temp: 0.7, Max: 150</i>"]
    Tech["<i>Technologies: vLLM, PyTorch, CUDA, Transformers</i>"]

    Input --> Model
    Model --> Inference
    Inference --> Streaming
    Streaming --> Output

    style LLM fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px,color:#000
    style Model fill:#e3f2fd,stroke:#0d47a1,stroke-width:1px,color:#000
    style Inference fill:#e3f2fd,stroke:#0d47a1,stroke-width:1px,color:#000
    style Streaming fill:#e3f2fd,stroke:#0d47a1,stroke-width:1px,color:#000
```

## Step 6: TTS Service (Audio Synthesis)

```mermaid
graph LR
    subgraph TTS["<b>TTS Service</b> <i>Port: 8003</i>"]
        direction LR
        TextProc["<b>Text Processing</b><br/><i>Text Analysis</i><br/><i>Emotion Detect</i><br/><i>Speed Control</i>"]
        AudioSynth["<b>Audio Synthesis</b><br/><i>MeloTTS</i><br/><i>Voice Cloning</i><br/><i>Quality Opt</i>"]
        ChunkStream["<b>Chunk Streaming</b><br/><i>100ms Chunks</i><br/><i>Base64 Encoded</i><br/><i>Real-time</i>"]
    end

    Input["<i>Input: Generated Text + Voice Parameters</i>"]
    Output["<i>Output: High-Quality Audio Chunks (Streaming)</i>"]
    Config["<i>Config: 22,050 Hz, 64 kbps, WAV, 100ms chunks, Speed 1.0x</i>"]
    Models["<i>Models: MeloTTS (Primary), gTTS (Fallback)</i>"]
    Tech["<i>Technologies: MeloTTS, PyTorch, torchaudio, pydub, webrtcvad</i>"]

    Input --> TextProc
    TextProc --> AudioSynth
    AudioSynth --> ChunkStream
    ChunkStream --> Output

    style TTS fill:#fce4ec,stroke:#880e4f,stroke-width:2px,color:#000
    style TextProc fill:#e3f2fd,stroke:#0d47a1,stroke-width:1px,color:#000
    style AudioSynth fill:#e3f2fd,stroke:#0d47a1,stroke-width:1px,color:#000
    style ChunkStream fill:#e3f2fd,stroke:#0d47a1,stroke-width:1px,color:#000
```

## Step 7: Vector Database

```mermaid
graph LR
    subgraph Qdrant["<b>Qdrant Database</b> <i>Port: 6333</i>"]
        direction LR
        Collections["<b>Collections</b><br/><i>Document Storage</i><br/><i>Metadata Mgmt</i>"]
        VectorOps["<b>Vector Operations</b><br/><i>Embedding Indexing</i><br/><i>Vector CRUD</i><br/><i>Batch Ops</i>"]
        Similarity["<b>Similarity Search</b><br/><i>Cosine Similarity</i><br/><i>Top-K Results</i><br/><i>Filtering</i>"]
    end

    Purpose["<i>Purpose: Store and retrieve contextual information for RAG</i>"]
    Config["<i>Config: 1024-dim vectors, Cosine Similarity, HNSW Index, Persistent Storage</i>"]
    Tech["<i>Technologies: Qdrant, Rust, HTTP API, gRPC API</i>"]

    Collections --> VectorOps
    VectorOps --> Similarity

    style Qdrant fill:#e0f2f1,stroke:#004d40,stroke-width:2px,color:#000
    style Collections fill:#e3f2fd,stroke:#0d47a1,stroke-width:1px,color:#000
    style VectorOps fill:#e3f2fd,stroke:#0d47a1,stroke-width:1px,color:#000
    style Similarity fill:#e3f2fd,stroke:#0d47a1,stroke-width:1px,color:#000
```

## Complete End-to-End Data Flow

```mermaid
graph LR
    UserInput["<b>User Input</b><br/><i>Hi there!</i>"]

    Frontend1["<b>Frontend App</b><br/><i>Port: 8080</i><br/><i>Input validation</i><br/><i>UI updates</i><br/><i>WebSocket connection</i>"]

    Orchestration["<b>Orchestration</b><br/><i>Port: 8000</i><br/><i>Session management</i><br/><i>History storage</i><br/><i>Context prep</i><br/><i>Pipeline coordination</i>"]

    RAG["<b>RAG Service</b><br/><i>Port: 8004</i><br/><i>BGE-Large-EN-v1.5</i><br/><i>Qdrant vector search</i><br/><i>BGE-Reranker</i><br/><i>Context retrieval</i>"]

    LLM["<b>LLM Service</b><br/><i>Port: 8002</i><br/><i>LLaMA-3-8B-Instruct</i><br/><i>AWQ quantized</i><br/><i>vLLM inference</i><br/><i>Token streaming</i>"]

    TTS["<b>TTS Service</b><br/><i>Port: 8003</i><br/><i>MeloTTS</i><br/><i>22,050 Hz, 64 kbps</i><br/><i>WAV format</i><br/><i>100ms chunks</i>"]

    Frontend2["<b>Frontend App</b><br/><i>Port: 8080</i><br/><i>HTML5 audio playback</i><br/><i>Real-time processing</i><br/><i>UI updates</i><br/><i>History update</i>"]

    AudioOutput["<b>Audio Output</b><br/><i>Hi there! It's nice</i><br/><i>to chat with you...</i>"]

    UserInput -->|WebSocket| Frontend1
    Frontend1 -->|WebSocket Message| Orchestration
    Orchestration -->|HTTP Request| RAG
    RAG -->|HTTP Request| LLM
    LLM -->|HTTP Request| TTS
    TTS -->|WebSocket Stream| Frontend2
    Frontend2 --> AudioOutput

    style UserInput fill:#e1f5ff,stroke:#01579b,stroke-width:2px,color:#000
    style Frontend1 fill:#fff3e0,stroke:#e65100,stroke-width:2px,color:#000
    style Frontend2 fill:#fff3e0,stroke:#e65100,stroke-width:2px,color:#000
    style Orchestration fill:#fff3e0,stroke:#e65100,stroke-width:3px,color:#000
    style RAG fill:#fff9c4,stroke:#f57f17,stroke-width:2px,color:#000
    style LLM fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px,color:#000
    style TTS fill:#fce4ec,stroke:#880e4f,stroke-width:2px,color:#000
    style AudioOutput fill:#e1f5ff,stroke:#01579b,stroke-width:2px,color:#000
```

## Performance Metrics & Timing

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        PERFORMANCE METRICS & TIMING                           │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  RESPONSE TIMING BREAKDOWN:                                                  │
│  • Connection initialization: ~360ms                                            │
│  • Session management: ~50ms                                                   │
│  • Context preparation: ~100ms                                                  │
│  • RAG retrieval: ~200ms                                                       │
│  • LLM generation: ~2,000ms (streaming)                                        │
│  • TTS synthesis: ~1,500ms (streaming)                                        │
│  • Audio playback: ~4,000ms (100+ chunks)                                     │
│  • Total end-to-end: ~4,600ms                                                  │
│                                                                                 │
│  THROUGHPUT CAPABILITIES:                                                   │
│  • LLM: 50+ tokens/second (vLLM optimized)                                     │
│  • TTS: 100+ chunks/second (MeloTTS streaming)                                │
│  • RAG: 1000+ queries/second (Qdrant vector search)                          │
│  • WebSocket: 1000+ messages/second (real-time)                              │
│                                                                                 │
│  RESOURCE UTILIZATION:                                                      │
│  • GPU Memory: ~8GB (LLM + TTS)                                               │
│  • CPU Usage: ~40% (orchestration + RAG)                                      │
│  • RAM Usage: ~16GB (all services)                                            │
│  • Network: ~1MB/s (audio streaming)                                          │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Technology Stack Summary

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            TECHNOLOGY STACK SUMMARY                            │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  FRONTEND:                                                                      │
│  • HTML5, JavaScript, WebSocket API, WebRTC API                               │
│  • Real-time UI updates, audio playback, connection management                │
│                                                                                 │
│  BACKEND SERVICES:                                                              │
│  • Python, FastAPI, AsyncIO, httpx                                            │
│  • Docker containerization, health monitoring                                 │
│                                                                                 │
│  AI MODELS:                                                                     │
│  • LLaMA-3-8B-Instruct (AWQ quantized) - LLM                                  │
│  • BGE-Large-EN-v1.5 - English embeddings                                     │
│  • multilingual-E5-Large - Multilingual embeddings                           │
│  • BGE-Reranker-Large - Context reranking                                     │
│  • MeloTTS - Neural text-to-speech                                             │
│                                                                                 │
│  INFRASTRUCTURE:                                                                │
│  • vLLM - High-throughput LLM inference                                       │
│  • Qdrant - Vector database                                                    │
│  • Docker Compose - Service orchestration                                     │
│  • GPU acceleration (CUDA)                                                     │
│                                                                                 │
│  COMMUNICATION:                                                                 │
│  • WebSocket - Real-time bidirectional communication                          │
│  • HTTP/REST - Service-to-service communication                               │
│  • WebRTC - Ultra-low latency voice communication                             │
└─────────────────────────────────────────────────────────────────────────────────┘
```

This end-to-end workflow diagram shows the complete journey from user text input to audio output, including all service names, models used, communication protocols, and performance metrics. The system is designed for production-grade conversational AI with real-time streaming and high-quality audio output.
