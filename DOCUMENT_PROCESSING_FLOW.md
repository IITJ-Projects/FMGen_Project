# End-to-End Document Processing Flow

## Complete Workflow: Upload → Extract → Chunk → Embed → Store

### Overview Diagram
```
Frontend UI → Upload Endpoint → File Storage
                  ↓
              Process Button → Process Endpoint
                  ↓
        ┌─────────────────────────────────┐
        │  STEP 1: Text Extraction       │
        │  (PDF/DOCX/TXT → Plain Text)    │
        └─────────────────────────────────┘
                  ↓
        ┌─────────────────────────────────┐
        │  STEP 2: Text Chunking           │
        │  (Split into 1024 char chunks)   │
        └─────────────────────────────────┘
                  ↓
        ┌─────────────────────────────────┐
        │  STEP 3: Document Structuring   │
        │  (Create UUID + Metadata)       │
        └─────────────────────────────────┘
                  ↓
        ┌─────────────────────────────────┐
        │  STEP 4: RAG Service Ingestion  │
        │  (Generate Embeddings + Store)  │
        └─────────────────────────────────┘
                  ↓
            Qdrant Vector DB
```

---

## Step-by-Step Detailed Flow

### **STEP 0: File Upload** 📤

**Location:** `rag_ingestion_app/main.py:726` - `upload_file()`

**What Happens:**
1. User selects PDF/DOCX/TXT file in UI
2. Frontend sends `POST /upload` with file data
3. Server validates:
   - File type (must be .pdf, .docx, or .txt)
   - File size (max 50MB)
4. Generates unique UUID for file: `file_id = uuid.uuid4()`
5. Saves file to disk: `uploads/{file_id}_{original_filename}`
6. Returns `file_id` to frontend

**Code:**
```python
# Generate unique file ID
file_id = str(uuid.uuid4())
file_path = UPLOAD_DIR / f"{file_id}_{file.filename}"

# Save uploaded file
async with aiofiles.open(file_path, 'wb') as f:
    content = await file.read()
    await f.write(content)
```

**Result:** File saved on disk, `file_id` returned to UI

---

### **STEP 1: Process Request** ▶️

**Location:** `rag_ingestion_app/main.py:781` - `process_file()`

**What Happens:**
1. User clicks "Process" button in UI
2. Frontend sends `POST /process/{file_id}`
3. Server finds file: `uploads/{file_id}_*`
4. Starts processing (all subsequent steps happen synchronously in this request)

---

### **STEP 2: Text Extraction** 📄

**Location:** `rag_ingestion_app/main.py:287` - `extract_text_from_file()`

**What Happens:**
1. Based on file extension, calls appropriate extractor:
   - **PDF:** Uses `PyPDF2` to extract text from each page
   - **DOCX:** Uses `python-docx` to extract paragraphs
   - **TXT:** Simple file read
2. Runs in thread pool (`asyncio.to_thread()`) to avoid blocking
3. Returns plain text string

**Code:**
```python
# Extract text from file (run in thread pool to avoid blocking)
text = await asyncio.to_thread(extract_text_from_file, file_path)
```

**Example Output:**
```
"This is page 1 content... This is page 2 content..."
```

**Result:** Plain text extracted from document

---

### **STEP 3: Text Chunking** ✂️

**Location:** `rag_ingestion_app/main.py:304` - `chunk_text()`

**What Happens:**
1. Text is split into chunks of **1024 characters** each
2. Chunks have **128 character overlap** (for context continuity)
3. Uses semantic boundary detection:
   - Prefers breaking at paragraphs
   - Then sentence boundaries
   - Then word boundaries
4. Runs in thread pool with timeout protection
5. If timeout, falls back to simple chunking (just split by size)

**Code:**
```python
# Use asyncio.wait_for with thread pool to prevent hanging
chunks = await asyncio.wait_for(
    asyncio.to_thread(chunk_text, text, CHUNK_SIZE, CHUNK_OVERLAP),
    timeout=timeout
)
```

**Example Output:**
```
[
  "This is chunk 1 (first 1024 chars)...",
  "...overlap (last 128) + new content (896 chars)...",
  "...overlap (last 128) + new content (896 chars)..."
]
```

**Result:** List of text chunks (typically 5-20 chunks for small PDF)

---

### **STEP 4: Document Structure Creation** 📝

**Location:** `rag_ingestion_app/main.py:527` - `generate_embeddings_for_chunks()`

**What Happens:**
1. For each chunk, creates a document structure:
   - **Generates UUID** for chunk (required by Qdrant)
   - **Text:** The chunk content
   - **Metadata:** file_id, chunk_index, total_chunks, chunk_size, source
2. **Important:** Embeddings are NOT generated here - only document structure
3. This is just preparing the data for the RAG service

**Code:**
```python
for i, chunk in enumerate(chunks):
    chunk_id = str(uuid.uuid4())  # Generate UUID for Qdrant
    document = {
        "id": chunk_id,
        "text": chunk,
        "metadata": {
            "file_id": file_id,
            "chunk_index": i + 1,
            "total_chunks": len(chunks),
            "chunk_size": len(chunk),
            "source": "file_upload"
        }
    }
    documents.append(document)
```

**Example Output:**
```json
[
  {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "text": "This is chunk 1...",
    "metadata": {
      "file_id": "1817ab3b-cf1d-4708-91a4-17344fae2344",
      "chunk_index": 1,
      "total_chunks": 10,
      "chunk_size": 1024,
      "source": "file_upload"
    }
  },
  ...
]
```

**Result:** List of document structures ready for ingestion

---

### **STEP 5: Batch Ingestion to RAG Service** 🚀

**Location:** `rag_ingestion_app/main.py:568` - `ingest_documents_to_rag()`

**What Happens:**
1. Documents are split into batches of 50 (to prevent memory issues)
2. Each batch is sent to RAG service via HTTP POST
3. Small delay (0.1s) between batches to yield control to event loop
4. Progress logged per batch

**Code:**
```python
# For small batches (< 50 docs), send all at once
if total_docs <= INGESTION_BATCH_SIZE:
    return await _ingest_batch(client, documents, 0, total_docs, total_docs)

# For larger batches, process in chunks
for batch_start in range(0, total_docs, INGESTION_BATCH_SIZE):
    batch = documents[batch_start:batch_end]
    result = await _ingest_batch(client, batch, ...)
    await asyncio.sleep(INGESTION_BATCH_DELAY)  # Yield control
```

**Request to RAG Service:**
```http
POST http://rag-service:8004/ingest
Content-Type: application/json

{
  "documents": [
    {
      "id": "uuid-1",
      "text": "chunk 1...",
      "metadata": {...}
    },
    ...
  ]
}
```

**Result:** HTTP request sent to RAG service

---

### **STEP 6: RAG Service Processing** 🧠

**Location:** `rag_service/main.py:580` - `ingest_documents()`

**What Happens:**

#### 6a. **Language Detection**
- For each document, detects language (English vs other)
- Uses `langdetect` library

#### 6b. **Embedding Generation**
- **English documents:** Uses `BAAI/bge-large-en-v1.5` model (1024 dimensions)
- **Multilingual documents:** Uses `intfloat/multilingual-e5-large` model (1024 dimensions)
- Generates vector embedding for each chunk's text

**Code:**
```python
# Detect language for each document
doc_language = detect_language(doc.text)

# Get appropriate embedding model
embedding_model = get_embedding_model(doc_language)

# Generate embedding (1024-dimensional vector)
embedding = embedding_model.encode(doc.text).tolist()
```

**Example Output:**
```
Embedding: [0.123, -0.456, 0.789, ..., 1024 dimensions total]
```

#### 6c. **Prepare for Qdrant**
- Combines document text, embedding vector, and metadata
- Adds language and embedding model info to metadata

**Code:**
```python
doc_dict = {
    "id": doc.id,  # UUID
    "text": doc.text,
    "metadata": {
        **(doc.metadata or {}),
        "language": doc_language,
        "embedding_model": "BGE" if doc_language == 'en' else "multilingual-E5"
    }
}
```

**Result:** Documents + embeddings ready for storage

---

### **STEP 7: Store in Qdrant** 💾

**Location:** `rag_service/main.py:633` - `upsert_documents()`

**What Happens:**
1. Connects to Qdrant database (running in separate container)
2. For each document:
   - Creates a "point" with:
     - **ID:** UUID (the chunk ID)
     - **Vector:** 1024-dimensional embedding
     - **Payload:** Text + metadata
3. Batch upserts all points to Qdrant collection
4. Qdrant stores vectors in its vector database for fast similarity search

**Code:**
```python
points = []
for doc, embedding in zip(documents, embeddings):
    point = {
        "id": doc["id"],  # UUID
        "vector": embedding,  # 1024-dimensional vector
        "payload": {
            "text": doc["text"],
            "metadata": doc.get("metadata", {})
        }
    }
    points.append(point)

# Upsert to Qdrant
await client.put(f"{base_url}/collections/documents/points", json={"points": points})
```

**Storage Structure:**
```
Qdrant Collection: "documents"
├── Point 1
│   ├── ID: "uuid-1"
│   ├── Vector: [0.123, -0.456, ...] (1024 dims)
│   └── Payload: {text: "...", metadata: {...}}
├── Point 2
│   ├── ID: "uuid-2"
│   ├── Vector: [0.234, -0.567, ...] (1024 dims)
│   └── Payload: {text: "...", metadata: {...}}
└── ...
```

**Result:** All chunks stored in Qdrant vector database

---

### **STEP 8: Response Back to Frontend** ✅

**What Happens:**
1. RAG service returns ingestion result:
   ```json
   {
     "ingested_count": 10,
     "failed_count": 0,
     "processing_time": 12.5
   }
   ```
2. RAG Ingestion App returns processing result:
   ```json
   {
     "file_id": "1817ab3b-...",
     "chunks_created": 10,
     "embeddings_generated": 10,
     "processing_time": 15.2,
     "status": "completed"
   }
   ```
3. Frontend displays success message

---

## Data Flow Summary

```
1. PDF File (upload)
   ↓
2. Plain Text (extraction)
   ↓
3. Text Chunks (chunking)
   ↓
4. Document Structures (UUID + metadata)
   ↓
5. HTTP POST to RAG Service
   ↓
6. Language Detection + Embedding Generation
   ↓
7. Vector + Text + Metadata
   ↓
8. Qdrant Vector Database Storage
```

## Key Components

- **RAG Ingestion App** (`rag-ingestion-service:8005`): Handles upload, extraction, chunking
- **RAG Service** (`rag-service:8004`): Generates embeddings, stores in Qdrant
- **Qdrant** (`qdrant-db:6333`): Vector database for storage and search

## Timing (Small PDF ~10 chunks)

- Upload: < 1 second
- Text Extraction: < 1 second
- Chunking: < 1 second
- Document Structuring: < 1 second
- Ingestion to RAG: 5-15 seconds (embedding generation takes time)
- Total: ~10-20 seconds

## Safety Features

1. **Async/Non-blocking:** All I/O operations use async/await
2. **Thread Pools:** CPU-bound tasks (extraction, chunking) run in threads
3. **Batching:** Large documents split into batches of 50
4. **Timeouts:** Chunking has timeout protection
5. **Error Recovery:** Failed batches don't stop entire process

