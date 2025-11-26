# Testing Small Document Processing

## Quick Test Guide

### Step 1: Prepare a Small Test PDF

- Create or find a small PDF (1-2 pages, ~100-500 KB)
- Example: A simple text document with 1-2 paragraphs
- Make sure it has extractable text (not just images)

### Step 2: Verify Services are Running

```bash
docker compose ps | grep -E "(rag-ingestion|rag-service|qdrant)"
```

All should show status: `Up` and `healthy`

### Step 3: Monitor Logs (Terminal 1)

Keep this running to watch the processing:

```bash
docker compose logs -f rag-ingestion-service rag-service
```

### Step 4: Access the UI

1. Open browser: `http://YOUR_SERVER_IP:8005`
2. Check health indicators:
   - RAG Service: Should show "healthy" (green)
   - LLM Service: Should show "healthy" (green)

### Step 5: Upload and Process

1. Upload the small PDF via drag-and-drop or browse
2. Click "Process" button on the uploaded file
3. Watch the logs for:
   ```
   ✅ Extracting text from...
   ✅ Chunking text from... (text length: X characters)
   ✅ Created X chunks from...
   📝 STEP 1/3: Generating document structures...
   🚀 STEP 2/3: Ingesting documents to RAG service...
   ✅ STEP 2/3: Ingestion completed...
   Processing completed...
   ```

### Step 6: Verify in RAG Service

Check document count:

```bash
curl http://YOUR_SERVER_IP:8004/documents/count
```

Test retrieval:

```bash
curl -X POST http://YOUR_SERVER_IP:8004/retrieve \
  -H "Content-Type: application/json" \
  -d '{
    "query": "test query from your document",
    "top_k": 5,
    "use_reranker": true,
    "language": "en"
  }'
```

## Expected Behavior (Small Document)

### Timing:

- **Text Extraction**: < 1 second
- **Chunking**: < 1 second (for small doc)
- **Document Structure Creation**: < 1 second
- **RAG Ingestion**: 5-15 seconds (depends on chunk count)
- **Total**: 10-20 seconds for a small document

### Logs to Watch For:

- No timeout warnings
- No blocking/hanging
- Progress messages every 10 chunks
- Success confirmation messages

## Troubleshooting

### If Processing Hangs:

1. Check logs for errors
2. Verify RAG service is responding:
   ```bash
   curl http://YOUR_SERVER_IP:8004/health
   ```
3. Check Qdrant is accessible:
   ```bash
   docker compose logs qdrant-db | tail -20
   ```

### If Chunking Times Out:

- Check text length in logs
- For very small files, chunking should be instant
- If it times out, the simple_chunk_text fallback should kick in

### If Ingestion Fails:

- Check UUID format errors
- Verify RAG service URL is correct
- Check network connectivity between services

## Next Steps After Successful Test

Once small document works:

1. Test with medium document (5-10 pages)
2. Test with larger document (20+ pages)
3. Monitor resource usage (CPU, memory)
4. Add batching for large documents (if needed)
