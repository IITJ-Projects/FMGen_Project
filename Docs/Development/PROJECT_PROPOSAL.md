## Zevo-AI: Production-Grade Multilingual Voice Agent (Project Proposal)

### Overview

Zevo-AI is a modular, containerized, real-time voice agent that understands multilingual speech, retrieves context-aware knowledge, reasons with a foundation model, and responds with natural-sounding speech. The system targets reliable low latency per stage, robust retrieval-augmented generation (RAG), and persistent short/long-term memory suitable for real-world assistants.

### Motivation and Problem Statement

Modern voice assistants often struggle with domain grounding, multilingual coverage, and smooth, low-latency interaction. Zevo-AI addresses these gaps by combining streaming ASR, dual-embedding RAG with re-ranking, a quantized instruction-tuned LLM, and streaming TTS, orchestrated in a pipeline designed for production reliability.

### Objectives

- **Real-time interaction**: Enable interactive, streaming, turn-by-turn conversations under tight latency budgets.
- **Multilingual support**: Handle English and additional languages with robust ASR, retrieval, and generation.
- **Grounded responses**: Use RAG with strong retrieval precision/recall and citation-like traceability.
- **Persistent memory**: Maintain short-term (session) and long-term (cross-session) memories to personalize interactions.
- **Production readiness**: Containerized microservices, observable, scalable, and testable.

### System Architecture (High-Level)

- **Orchestration**: Vocode pipeline managing streaming flow: ASR → RAG → LLM → TTS.
- **ASR Service**: Streaming `faster-whisper-medium` with partial hypotheses for low-latency transcripts.
- **RAG Service**: Qdrant vector DB with dual embeddings (English: BGE; Multilingual: E5) plus a reranker; includes ingestion app.
- **LLM Service**: `LLaMA-3-8B-Instruct` (AWQ-quantized) hosted via vLLM for fast token throughput and batching.
- **Memory**: Vocode caching for short-term; Qdrant for long-term episodic and semantic memory.
- **TTS Service**: Streaming MeloTTS producing natural audio with incremental playback.
- **Containerization**: Each component as a microservice orchestrated via Docker Compose.

### Data Flow (Streaming)

1. Audio frames captured and sent to ASR.
2. Streaming transcripts (partials → finalized) sent to orchestration.
3. RAG retrieves and reranks relevant passages (dual-embedding + reranker) based on language.
4. LLM conditions on user input + retrieved context + memory.
5. Generated tokens streamed to TTS; TTS streams audio back to client for immediate playback.

### Key Innovations

- **Dual-embedding RAG**: BGE for English and multilingual-E5 for non-English queries, improving cross-lingual retrieval.
- **Memory-aware generation**: Combining Vocode’s short-term cache with Qdrant-backed long-term knowledge to reduce repetition and improve continuity.
- **Fully streaming pipeline**: Partial ASR → token streaming → incremental TTS for minimal perceived latency.

### Datasets and Knowledge Sources

- **Seed knowledge**: Course materials, FAQs, selected documentation; uploaded via the ingestion app (`rag_ingestion_app`).
- **Public datasets**: Optional for evaluation (Common Voice for ASR; multilingual QA datasets for RAG/LLM retrieval testing).
- **User-provided docs**: Support PDFs, text, and web content via ingestion UI.

### Evaluation and Metrics

- **ASR**: Word Error Rate (WER) and partial-to-final stability; streaming latency per chunk.
- **RAG**: Recall@k, MRR, and nDCG on a labeled query set; reranker ablations.
- **LLM**: Response groundedness (human eval rubric), hallucination rate, task success rate.
- **End-to-end latency**: ASR first-token time, LLM TTFT, TTS first-audio latency; 95th percentile targets.
- **TTS**: Mean Opinion Score (MOS) via small-scale human study; intelligibility and prosody.
- **Memory**: Correct recall of session facts and cross-session entities; read/write latency to Qdrant.

### Milestones and Timeline (6–8 Weeks Example)

- **Week 1**: Baseline services scaffolded; Docker Compose up; simple E2E stub.
- **Week 2**: Streaming ASR integrated; initial RAG with single embedding; ingestion app usable.
- **Week 3**: Dual embeddings + reranker; add evaluation harness for retrieval metrics.
- **Week 4**: vLLM deployment with AWQ model; prompt templates; grounding with RAG context.
- **Week 5**: Streaming MeloTTS integration; end-to-end streaming path validated; basic memory.
- **Week 6**: Long-term memory in Qdrant; session memory tuning; latency profiling and optimization.
- **Week 7**: Multilingual evaluation; robustness tests; guardrails and fallback strategies.
- **Week 8**: Final polish; documentation; demo and report.

### Risks and Mitigations

- **Latency spikes**: Use streaming everywhere, quantized LLM on vLLM, and batch-friendly configs; monitor p95/p99, enable backpressure.
- **Hallucinations**: Enforce retrieval-first prompts, cite retrieved chunks, restrict system prompts, and optionally add rule-based guardrails.
- **Multilingual retrieval errors**: Route per-language embeddings; fallback to cross-lingual search; expand multilingual corpus.
- **ASR/TTS mismatch**: Align sample rates and chunk sizes; normalize text; add punctuation handling.
- **Operational complexity**: Keep services modular; provide scripts and health checks; add observability.

### Experimental Plan and Ablations

- Single vs dual embeddings for multilingual queries.
- With vs without reranker; compare Recall@k and nDCG.
- Quantized vs non-quantized model (where feasible) for quality vs latency.
- Memory on/off; evaluate personalization benefits and error reduction.

### Implementation Plan (Codebase Mapping)

- `asr_service/`: Streaming Whisper server with partial hypotheses.
- `rag_service/`: Qdrant client, dual-embedding indexers, reranker, retrieval API.
- `rag_ingestion_app/`: UI + pipeline for document upload, chunking, language detection, and indexing.
- `llm_service/`: vLLM server wrapper, prompt templates, context assembly.
- `tts_service/`: Streaming MeloTTS gateway.
- `orchestration_service/`: Vocode pipeline glue for ASR → RAG → LLM → TTS.
- `docker-compose.yml`: Service wiring, networks, and environment configuration.

### Ethical and Safety Considerations

- Respect privacy: avoid storing raw audio by default; redact PII in logs.
- Transparency: communicate limitations; prefer grounded answers; escalate uncertain cases.
- Bias and fairness: evaluate multilingual parity; include diverse test cases.

### Resources and Budget (Class-Scale)

- Single GPU (12–24 GB) for vLLM with AWQ; CPU assist for ASR/TTS if needed.
- Qdrant (Docker) with SSD-backed storage.
- Modest dataset curation for evaluation; small participant pool for MOS.

### Deliverables

- Working demo with end-to-end streaming voice conversation.
- Evaluation report with metrics (WER, retrieval, latency, MOS, groundedness).
- Deployment guide and architecture docs; `docker-compose.yml` for reproducibility.
- Screencast/video demo and in-class presentation.

### Success Criteria

- p95 end-to-end (speech-in to first audio-out) under a practical threshold for interactive use.
- Significant retrieval gains from dual embeddings and reranking.
- Demonstrably reduced hallucinations via grounding and memory.
- Functional multilingual support across at least two non-English languages.

### References (Representative)

- Vocode pipeline for real-time conversational orchestration.
- Qdrant vector DB for scalable semantic search.
- vLLM for high-throughput LLM serving and AWQ quantization.
- faster-whisper for efficient streaming ASR; MeloTTS for streaming synthesis.
