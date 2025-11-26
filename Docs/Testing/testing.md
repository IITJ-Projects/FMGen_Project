### Zevo AI - Production Testing Guide

Base URL: `http://164.52.194.203`

Note: Append service ports as shown below. Add `-v` to curl for verbose output. If you have a gateway/ingress, adjust hosts/paths accordingly.

### Using Postman

- You can import any curl snippet below into Postman: Open Postman → Import → Raw text → paste the curl.
- Or import the provided Postman Collection and Environment JSON located in `postman/`.
  - Collection: `postman/ZevoAI.postman_collection.json`
  - Environment: `postman/ZevoAI.postman_environment.json`
  - After importing, select the environment and run requests directly.

Postman mappings (for quick setup):

- ASR health: GET `http://164.52.194.203:8001/health`
- ASR transcribe_text: POST `http://164.52.194.203:8001/transcribe_text`
  - Headers: `Content-Type: application/json`
  - Body (raw): `{ "text": "Quick ASR service test" }`
- ASR transcribe (file): POST `http://164.52.194.203:8001/transcribe`

  - Body: form-data → key `file` (type: File) → choose local audio file

- LLM health: GET `http://164.52.194.203:8002/health`
- LLM generate: POST `http://164.52.194.203:8002/generate`
  - Headers: `Content-Type: application/json`
  - Body (raw): `{ "prompt": "In one sentence, explain RAG.", "max_tokens": 64, "temperature": 0.2 }`
- LLM generate_stream (SSE): POST `http://164.52.194.203:8002/generate_stream`

  - Note: Postman may not render SSE progressively; use curl for live streaming.

- TTS health: GET `http://164.52.194.203:8003/health`
- TTS speak_status: POST `http://164.52.194.203:8003/speak_status`
  - Headers: `Content-Type: application/json`
  - Body (raw): `{ "text": "Testing TTS readiness" }`
- TTS speak: POST `http://164.52.194.203:8003/speak`

  - Headers: `Content-Type: application/json`
  - Body (raw): `{ "text": "Hello from production TTS", "sample_rate": 24000 }`

- RAG health: GET `http://164.52.194.203:8004/health`
- RAG embed: POST `http://164.52.194.203:8004/embed`
  - Headers: `Content-Type: application/json`
  - Body (raw): `["What is Zevo AI?","RAG pipelines are useful."]`
- RAG retrieve: POST `http://164.52.194.203:8004/retrieve`
  - Headers: `Content-Type: application/json`
  - Body (raw): `{ "query": "What does our agent do?", "top_k": 5, "use_reranker": true }`
- RAG ingest: POST `http://164.52.194.203:8004/ingest`

  - Headers: `Content-Type: application/json`
  - Body (raw): `{ "documents": [{ "id": "doc-1", "text": "Zevo is a multilingual voice agent.", "metadata": {"source": "seed"} }, { "id": "doc-2", "text": "It uses Qdrant for vector search.", "metadata": {"source": "seed"} }] }`

- Qdrant health: GET `http://164.52.194.203:6333/healthz`
- Qdrant collections: GET `http://164.52.194.203:6333/collections`

- Orchestration health: GET `http://164.52.194.203:8000/health`
- Orchestration status: GET `http://164.52.194.203:8000/status`

### ASR Service (faster-whisper) - Port 8001

- Health

```bash
curl -sS http://164.52.194.203:8001/health | jq .
```

- Quick test without audio

```bash
curl -sS -X POST http://164.52.194.203:8001/transcribe_text \
  -H "Content-Type: application/json" \
  -d '{"text":"Quick ASR service test"}' | jq .
```

- Audio upload (replace path with a local sample)

```bash
curl -sS -X POST http://164.52.194.203:8001/transcribe \
  -H "Expect:" \
  -F "file=@/path/to/sample.wav;type=audio/wav" | jq .
```

### LLM Service (vLLM LLaMA-3-8B) - Port 8002

- Health

```bash
curl -sS http://164.52.194.203:8002/health | jq .
```

- Non-streaming generation

```bash
curl -sS -X POST http://164.52.194.203:8002/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "In one sentence, explain RAG.",
    "max_tokens": 64,
    "temperature": 0.2
  }' | jq .
```

- Streaming generation (SSE)

```bash
curl -N -sS -X POST http://164.52.194.203:8002/generate_stream \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Say hello.","max_tokens":32,"temperature":0.7}'
```

### TTS Service (MeloTTS) - Port 8003

- Health

```bash
curl -sS http://164.52.194.203:8003/health | jq .
```

- Readiness check

```bash
curl -sS -X POST http://164.52.194.203:8003/speak_status \
  -H "Content-Type: application/json" \
  -d '{"text":"Testing TTS readiness"}' | jq .
```

- Synthesize to file

```bash
curl -sS -X POST http://164.52.194.203:8003/speak \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello from production TTS","sample_rate":24000}' \
  --output speech_output.bin && file speech_output.bin
```

### RAG Service - Port 8004

- Health

```bash
curl -sS http://164.52.194.203:8004/health | jq .
```

- Embed check

```bash
curl -sS -X POST http://164.52.194.203:8004/embed \
  -H "Content-Type: application/json" \
  -d '["What is Zevo AI?","RAG pipelines are useful."]' | jq .
```

- Retrieve (requires ingested data)

```bash
curl -sS -X POST http://164.52.194.203:8004/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query":"What does our agent do?","top_k":5,"use_reranker":true}' | jq .
```

- Optional: ingest quick seed docs

```bash
curl -sS -X POST http://164.52.194.203:8004/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "documents":[
      {"id":"doc-1","text":"Zevo is a multilingual voice agent.","metadata":{"source":"seed"}},
      {"id":"doc-2","text":"It uses Qdrant for vector search.","metadata":{"source":"seed"}}
    ]
  }' | jq .
```

### Qdrant Vector DB - Port 6333

- Health

```bash
curl -sS http://164.52.194.203:6333/healthz | jq .
```

- List collections

```bash
curl -sS http://164.52.194.203:6333/collections | jq .
```

### Orchestration Service (Vocode) - Port 8000

- Health (if implemented)

```bash
curl -sS http://164.52.194.203:8000/health | jq .
```

- Status (optional; may vary)

```bash
curl -sS http://164.52.194.203:8000/status | jq . || true
```

### RAG Ingestion Web App - Port 8005

- UI response headers

```bash
curl -sS -I http://164.52.194.203:8005
```

- Health (if implemented)

```bash
curl -sS http://164.52.194.203:8005/health | jq . || true
```

### Optional end-to-end smoke (RAG → LLM)

```bash
CTX=$(curl -sS -X POST http://164.52.194.203:8004/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query":"Summarize our architecture.","top_k":3,"use_reranker":true}' \
  | jq -r '[.documents[].text] | join("\n---\n")')

curl -sS -X POST http://164.52.194.203:8002/generate \
  -H "Content-Type: application/json" \
  -d "$(jq -n --arg p 'Summarize our architecture' --arg c "$CTX" '{prompt:$p, context:$c, max_tokens:128, temperature:0.3}')" \
  | jq .
```

### Troubleshooting

- If a service returns 503, it may still be loading models; retry after a minute.
- Ensure GPU is available for the LLM service if required by your deploy.
- For RAG issues, confirm Qdrant health and `qdrant_connected: true` in RAG health.
- If using TLS/ingress, replace direct port access with the correct public URL.
