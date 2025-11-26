# Zevo AI - End-to-End Voice & Text Processing Architecture

## 🏗️ System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           ZEVO AI CONVERSATIONAL PLATFORM                        │
│                    Production-Grade Multilingual Voice Agent                      │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## 📱 Frontend Layer (User Interface)

### Dual-Mode Interface

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                FRONTEND LAYER                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐  │
│  │   TEXT CHAT MODE    │    │   VOICE CALL MODE   │    │   MODE TOGGLE       │  │
│  │                     │    │                     │    │                     │  │
│  │ • Text Input        │    │ • Voice Recording   │    │ • Switch between    │  │
│  │ • Send Button       │    │ • Call Controls     │    │   text/voice       │  │
│  │ • Chat History      │    │ • Mute/Hold/End     │    │ • Real-time status  │  │
│  │ • Typing Indicator  │    │ • Call Duration     │    │ • Connection status │  │
│  └─────────────────────┘    └─────────────────────┘    └─────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## 🔄 Processing Pipeline Architecture

### 1. TEXT MODE FLOW

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              TEXT PROCESSING FLOW                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  User Types Message                                                             │
│         │                                                                       │
│         ▼                                                                       │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐            │
│  │   Frontend      │    │   WebSocket     │    │  Orchestration  │            │
│  │   (main.js)     │───▶│   Connection    │───▶│   Service      │            │
│  │                 │    │   (Real-time)   │    │   (FastAPI)    │            │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘            │
│         │                         │                         │                  │
│         │                         │                         ▼                  │
│         │                         │                ┌─────────────────┐            │
│         │                         │                │   RAG Service  │            │
│         │                         │                │   (Qdrant)     │            │
│         │                         │                │   • Context    │            │
│         │                         │                │   • Memory     │            │
│         │                         │                │   • Search     │            │
│         │                         │                └─────────────────┘            │
│         │                         │                         │                  │
│         │                         │                         ▼                  │
│         │                         │                ┌─────────────────┐            │
│         │                         │                │   LLM Service   │            │
│         │                         │                │   (vLLM)        │            │
│         │                         │                │   • LLaMA-3-8B  │            │
│         │                         │                │   • Streaming  │            │
│         │                         │                │   • AWQ Quant.  │            │
│         │                         │                └─────────────────┘            │
│         │                         │                         │                  │
│         │                         │                         ▼                  │
│         │                         │                ┌─────────────────┐            │
│         │                         │                │   TTS Service   │            │
│         │                         │                │   (MeloTTS)     │            │
│         │                         │                │   • Audio Gen.  │            │
│         │                         │                │   • WAV Format  │            │
│         │                         │                │   • Streaming  │            │
│         │                         │                └─────────────────┘            │
│         │                         │                         │                  │
│         │                         │                         ▼                  │
│         │                         │                ┌─────────────────┐            │
│         │                         │                │   Response      │            │
│         │                         │                │   • Text       │            │
│         │                         │                │   • Audio      │            │
│         │                         │                │   • Latency    │            │
│         │                         │                └─────────────────┘            │
│         │                         │                         │                  │
│         │                         │                         ▼                  │
│         │                         │                ┌─────────────────┐            │
│         │                         │                │   Frontend      │            │
│         │                         │                │   Display      │            │
│         │                         │                │   • Text Stream │            │
│         │                         │                │   • Audio Play  │            │
│         │                         │                │   • UI Update  │            │
│         │                         │                └─────────────────┘            │
│         │                         │                         │                  │
│         └─────────────────────────┴─────────────────────────┘                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 2. VOICE MODE FLOW

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                             VOICE PROCESSING FLOW                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  User Speaks (Voice Call Mode)                                                 │
│         │                                                                       │
│         ▼                                                                       │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐            │
│  │   Frontend      │    │   WebRTC        │    │  Orchestration  │            │
│  │   (main.js)     │───▶│   Data Channel  │───▶│   Service      │            │
│  │   • VAD         │    │   (Ultra-low    │    │   (FastAPI)    │            │
│  │   • Recording   │    │    Latency)    │    │                │            │
│  │   • Auto-stop   │    │                │    │                │            │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘            │
│         │                         │                         │                  │
│         │                         │                         ▼                  │
│         │                         │                ┌─────────────────┐            │
│         │                         │                │   ASR Service   │            │
│         │                         │                │   (faster-     │            │
│         │                         │                │    whisper)     │            │
│         │                         │                │   • Streaming  │            │
│         │                         │                │   • Real-time  │            │
│         │                         │                │   • WebM→WAV   │            │
│         │                         │                └─────────────────┘            │
│         │                         │                         │                  │
│         │                         │                         ▼                  │
│         │                         │                ┌─────────────────┐            │
│         │                         │                │   RAG Service  │            │
│         │                         │                │   (Qdrant)      │            │
│         │                         │                │   • Context    │            │
│         │                         │                │   • Memory     │            │
│         │                         │                │   • Search     │            │
│         │                         │                └─────────────────┘            │
│         │                         │                         │                  │
│         │                         │                         ▼                  │
│         │                         │                ┌─────────────────┐            │
│         │                         │                │   LLM Service   │            │
│         │                         │                │   (vLLM)        │            │
│         │                         │                │   • Streaming  │            │
│         │                         │                │   • Tokens      │            │
│         │                         │                │   • Real-time  │            │
│         │                         │                └─────────────────┘            │
│         │                         │                         │                  │
│         │                         │                         ▼                  │
│         │                         │                ┌─────────────────┐            │
│         │                         │                │   TTS Service   │            │
│         │                         │                │   (MeloTTS)     │            │
│         │                         │                │   • Audio Gen.  │            │
│         │                         │                │   • WAV Format  │            │
│         │                         │                │   • Streaming   │            │
│         │                         │                │   • Chunks      │            │
│         │                         │                └─────────────────┘            │
│         │                         │                         │                  │
│         │                         │                         ▼                  │
│         │                         │                ┌─────────────────┐            │
│         │                         │                │   WebRTC       │            │
│         │                         │                │   Response      │            │
│         │                         │                │   • Audio       │            │
│         │                         │                │   • Text        │            │
│         │                         │                │   • Streaming   │            │
│         │                         │                └─────────────────┘            │
│         │                         │                         │                  │
│         │                         │                         ▼                  │
│         │                         │                ┌─────────────────┐            │
│         │                         │                │   Frontend      │            │
│         │                         │                │   Playback      │            │
│         │                         │                │   • HTML5 Audio│            │
│         │                         │                │   • Real-time   │            │
│         │                         │                │   • Continuous  │            │
│         │                         │                └─────────────────┘            │
│         │                         │                         │                  │
│         │                         │                         ▼                  │
│         │                         │                ┌─────────────────┐            │
│         │                         │                │   Auto-Listen   │            │
│         │                         │                │   • VAD         │            │
│         │                         │                │   • Continuous  │            │
│         │                         │                │   • Two-way     │            │
│         │                         │                └─────────────────┘            │
│         │                         │                         │                  │
│         └─────────────────────────┴─────────────────────────┘                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## 🔧 Service Layer Architecture

### Core Services

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              SERVICE LAYER                                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │  ASR Service    │  │  LLM Service    │  │  TTS Service    │  │ RAG Service│ │
│  │  (Port 8001)    │  │  (Port 8002)    │  │  (Port 8003)    │  │ (Port 8004)│ │
│  │                 │  │                 │  │                 │  │             │ │
│  │ • faster-       │  │ • vLLM Engine  │  │ • MeloTTS       │  │ • Qdrant DB │ │
│  │   whisper       │  │ • LLaMA-3-8B   │  │ • Streaming     │  │ • Dual      │ │
│  │ • Streaming     │  │ • AWQ Quant.   │  │ • WAV Format    │  │   Embeddings│ │
│  │ • Real-time     │  │ • Token Stream │  │ • Audio Chunks │  │ • Reranker  │ │
│  │ • WebM→WAV      │  │ • Fast Response│  │ • Voice Quality │  │ • Memory    │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  └─────────────┘ │
│                                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │ Orchestration   │  │  Frontend       │  │  Nginx Proxy    │  │  Qdrant DB  │ │
│  │ Service         │  │  (Port 80/443)  │  │  (Port 80/443)  │  │ (Port 6333)│ │
│  │ (Port 8000)     │  │                 │  │                 │  │             │ │
│  │                 │  │ • HTML/CSS/JS   │  │ • Load Balancer │  │ • Vector DB │ │
│  │ • FastAPI       │  │ • WebSocket     │  │ • SSL/TLS       │  │ • Milestones│ │
│  │ • WebSocket     │  │ • WebRTC        │  │ • CORS          │  │ • Context   │ │
│  │ • WebRTC        │  │ • Audio API    │  │ • Routing       │  │ • Search    │ │
│  │ • Pipeline      │  │ • VAD           │  │ • WebSocket     │  │ • Memory    │ │
│  │ • Latency Track │  │ • Dual Mode     │  │ • WebRTC        │  │ • RAG       │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## 🌐 Network & Communication Flow

### Protocol Stack

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            COMMUNICATION LAYERS                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │   HTTP/HTTPS    │  │   WebSocket     │  │     WebRTC      │  │   Internal  │ │
│  │   (REST API)    │  │   (Real-time)   │  │   (Ultra-low    │  │   Services  │ │
│  │                 │  │                 │  │    Latency)     │  │             │ │
│  │ • Health Check  │  │ • Text Chat     │  │ • Voice Call    │  │ • HTTP      │ │
│  │ • API Calls     │  │ • Streaming    │  │ • Data Channel  │  │ • gRPC      │ │
│  │ • File Upload   │  │ • LLM Tokens    │  │ • Audio Stream │  │ • Internal  │ │
│  │ • Status        │  │ • TTS Chunks    │  │ • ICE Candidates│  │ • Service   │ │
│  │ • Configuration │  │ • Real-time     │  │ • STUN/TURN     │  │ • Discovery │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## 📊 Data Flow & Latency Optimization

### Latency Tracking

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            LATENCY OPTIMIZATION                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │   ASR Step      │  │   RAG Step      │  │   LLM Step      │  │  TTS Step   │ │
│  │   (Target:      │  │   (Target:      │  │   (Target:      │  │  (Target:   │ │
│  │   <2s)          │  │   <500ms)       │  │   <3s)          │  │  <1s)       │ │
│  │                 │  │                 │  │                 │  │             │ │
│  │ • Streaming     │  │ • Vector Search │  │ • Token Stream  │  │ • Audio     │ │
│  │ • Real-time     │  │ • Context      │  │ • Streaming     │  │   Chunks    │ │
│  │ • VAD           │  │ • Memory       │  │ • Fast Response │  │ • WAV Format│ │
│  │ • WebM→WAV      │  │ • Reranking    │  │ • AWQ Quant.    │  │ • HTML5     │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  └─────────────┘ │
│                                                                                 │
│  Total Pipeline Latency: <6.5s (Target: <5s)                                   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## 🔄 Continuous Conversation Flow

### Voice Call Mode (Phone-like)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        CONTINUOUS CONVERSATION FLOW                           │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  Start Call → Listen → User Speaks → Auto-Process → AI Responds → Listen Again │
│      │         │          │            │              │            │           │
│      ▼         ▼          ▼            ▼              ▼            ▼           │
│  WebRTC    VAD/3s     ASR→Text    LLM→Tokens    TTS→Audio   Auto-Listen       │
│  Connect   Timer      Streaming    Streaming     Streaming   Continuous         │
│                                                                                 │
│  Features:                                                                      │
│  • Mute/Unmute     • Hold/Resume     • End Call     • Call Duration           │
│  • Two-way Talk    • Auto-Listen     • VAD         • Real-time Status         │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## 🛠️ Technology Stack

### Frontend Technologies

- **HTML5/CSS3**: Modern UI with Tailwind CSS
- **JavaScript**: ES6+ with WebSocket & WebRTC APIs
- **Web Audio API**: Advanced audio processing
- **WebRTC**: Ultra-low latency communication
- **VAD**: Voice Activity Detection

### Backend Technologies

- **FastAPI**: High-performance Python web framework
- **vLLM**: High-throughput LLM inference
- **MeloTTS**: High-quality text-to-speech
- **faster-whisper**: Optimized ASR
- **Qdrant**: Vector database for RAG
- **WebSocket**: Real-time communication
- **Docker**: Containerized services

### Infrastructure

- **Nginx**: Reverse proxy & load balancer
- **Docker Compose**: Service orchestration
- **SSL/TLS**: Secure communication
- **WebRTC**: P2P communication
- **STUN/TURN**: NAT traversal

## 🎯 Key Features

### Text Mode

- ✅ Real-time text chat
- ✅ Streaming LLM responses
- ✅ Audio playback
- ✅ Conversation history
- ✅ Latency monitoring

### Voice Mode

- ✅ Continuous voice calls
- ✅ Phone-like conversation
- ✅ Mute/Hold/End controls
- ✅ Auto-listening
- ✅ Two-way communication
- ✅ Call duration tracking

### Performance

- ✅ <6.5s total latency
- ✅ Real-time streaming
- ✅ Ultra-low latency WebRTC
- ✅ Optimized audio processing
- ✅ Memory management
- ✅ Error recovery

## 🔧 Configuration

### Environment Variables

```bash
# Service Ports
ASR_SERVICE_URL=http://asr-service:8001
LLM_SERVICE_URL=http://llm-service:8002
TTS_SERVICE_URL=http://tts-service:8003
RAG_SERVICE_URL=http://rag-service:8004

# WebRTC Configuration
WEBRTC_ENABLED=true
ICE_SERVERS=[{"urls": "stun:stun.l.google.com:19302"}]

# Audio Configuration
SAMPLE_RATE=16000
CHUNK_DURATION_MS=30
USE_OPUS=false
BITRATE=16
```

This architecture provides a complete, production-ready conversational AI platform with both text and voice capabilities, optimized for real-time performance and user experience.
