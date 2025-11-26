# Zevo AI Documentation

This folder contains comprehensive documentation for the Zevo AI project, organized by category.

## 📖 Documentation Index

### Architecture & Design
Detailed system architecture and design documents:

- **[Text Mode Architecture](Architecture/TEXT_MODE_ARCHITECTURE.md)** - Complete architecture for text mode with streaming LLM, RAG integration, and manual TTS
- **[End-to-End Architecture Flow](Architecture/ARCHITECTURE_FLOW.md)** - System-wide architecture covering both text and voice modes
- **[End-to-End Workflow](Architecture/END_TO_END_WORKFLOW.md)** - Step-by-step processing workflows with detailed diagrams
- **[Simple Data Flow Diagram](Architecture/SIMPLE_DATA_FLOW_DIAGRAM.md)** - High-level data flow visualization

### Deployment & Operations
Production deployment and operational guides:

- **[Deployment Guide](Deployment/DEPLOYMENT_GUIDE.md)** - Complete step-by-step server deployment instructions including Docker setup, NVIDIA configuration, and security
- **[Services Structure](Services/SERVICES_STRUCTURE.md)** - Detailed service architecture, endpoints, and communication patterns

### Development & Testing
Development guides and testing documentation:

- **[Testing Guide](Testing/testing.md)** - Production testing guide with API endpoint examples and Postman integration
- **[Project Proposal](Development/PROJECT_PROPOSAL.md)** - Original project proposal, objectives, and technical specifications

## 🔄 Recent Updates

### Text Mode Enhancements (Latest)
- **Real-Time Streaming**: Token-by-token LLM streaming for ChatGPT-like experience
- **Manual TTS**: Speaker icon for on-demand audio playback in text mode
- **Voice Input**: Support for voice input in text mode (ASR → text, no automatic TTS)
- **Markdown Rendering**: Formatted responses with proper Markdown display
- **Adaptive Responses**: Dynamic response length based on user intent (detailed vs concise)

### Voice Mode Features
- **Automatic TTS**: Real-time audio synthesis for voice conversations
- **Smooth Playback**: Optimized audio buffering and chunk throttling
- **WebRTC Support**: Ultra-low latency communication

## 📝 Quick Links

- [Main README](../README.md) - Project overview and quick start
- [RAG Ingestion App](../rag_ingestion_app/README.md) - RAG ingestion application guide

## 🔍 Document Status

All documents are kept up-to-date with the latest codebase. If you find outdated information, please update the relevant document or create an issue.

---
*Last Updated: Based on latest codebase with streaming enhancements and text mode improvements*
