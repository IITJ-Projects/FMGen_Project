# RAG Ingestion Application

A modern web application for uploading and processing documents for RAG (Retrieval-Augmented Generation) systems.

## Features

- **Modern Web UI**: Beautiful, responsive interface with drag-and-drop file upload
- **Multi-format Support**: Upload PDF, DOCX, and TXT files
- **Real-time Processing**: Watch as files are processed and embedded
- **Health Monitoring**: Live status of RAG and LLM services
- **File Management**: Upload, process, and delete files with ease
- **Progress Tracking**: Visual feedback for upload and processing operations

## Quick Start

1. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

2. **Start the Application**:

   ```bash
   python main.py
   ```

3. **Access the Web UI**:
   Open your browser and navigate to `http://localhost:8005`

## Usage

### Web Interface

1. **Upload Files**: Drag and drop files or click to browse

   - Supported formats: PDF, DOCX, TXT
   - Maximum file size: 50MB

2. **Process Files**: Click the "Process" button to:

   - Extract text from documents
   - Split into chunks
   - Generate embeddings
   - Store in RAG service

3. **Monitor Progress**: Watch real-time status updates and processing results

### API Endpoints

- `GET /` - Web UI
- `GET /health` - Service health check
- `POST /upload` - Upload a file
- `POST /process/{file_id}` - Process uploaded file
- `GET /files` - List uploaded files
- `DELETE /files/{file_id}` - Delete a file
- `POST /cleanup` - Manually trigger file cleanup
- `GET /storage-info` - Get storage usage information

## Configuration

The application connects to:

- **RAG Service**: `http://localhost:8004` (configurable via `RAG_SERVICE_URL`)
- **LLM Service**: `http://localhost:8002` (configurable via `LLM_SERVICE_URL`)

### Environment Variables

- `FILE_RETENTION_HOURS`: How long to keep uploaded files (default: 24 hours)
- `CLEANUP_INTERVAL_HOURS`: How often to run automatic cleanup (default: 6 hours)
- `MAX_UPLOAD_SIZE_MB`: Maximum file size for uploads (default: 50 MB)

### Chunking Strategy Configuration

The application uses a production-ready chunking strategy optimized for LLaMA-3-8B + Qdrant + Dual Embeddings:

- `CHUNK_SIZE`: Optimal chunk size in characters (default: 1024 = ~150-200 words)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 128 = 12.5% overlap)
- `MIN_CHUNK_SIZE`: Minimum chunk size to avoid noise (default: 256 chars)
- `MAX_CHUNK_SIZE`: Maximum chunk size for embedding models (default: 2048 chars)
- `SENTENCE_OVERLAP_RATIO`: Sentence overlap ratio (default: 0.3 = 30%)
- `PARAGRAPH_BREAK_WEIGHT`: Weight for paragraph breaks (default: 0.8)
- `HEADER_DETECTION`: Enable document header detection (default: true)
- `TABLE_HANDLING`: How to handle tables (default: "preserve")

## File Processing Pipeline

1. **Text Extraction**: Extract text from uploaded documents
2. **Chunking**: Split text into overlapping chunks (512 chars with 50 char overlap)
3. **Embedding Generation**: Create embeddings for each chunk
4. **RAG Ingestion**: Store documents in the RAG service for retrieval

## Development

### Project Structure

```
rag_ingestion_app/
├── main.py              # FastAPI application
├── requirements.txt     # Python dependencies
├── templates/          # HTML templates
│   └── index.html     # Main UI template
├── static/            # Static assets
└── uploads/           # Uploaded files (created automatically)
```

### Adding New File Types

To support additional file formats:

1. Add the file extension to the `allowed_extensions` set in `main.py`
2. Implement a new extraction function following the pattern of existing ones
3. Update the `extract_text_from_file` function to handle the new type

## Troubleshooting

### Common Issues

1. **Service Connection Errors**: Ensure RAG and LLM services are running
2. **File Upload Failures**: Check file size and format restrictions
3. **Processing Errors**: Verify file content is extractable text

### Health Check

The web UI displays real-time health status of connected services. Check the status indicators at the top of the page.

## Storage Management

The application includes automatic file cleanup to manage disk space:

- **Automatic Cleanup**: Files are automatically deleted after the retention period (default: 24 hours)
- **Manual Cleanup**: Use the web UI or API to trigger immediate cleanup
- **Storage Monitoring**: Real-time storage usage information in the web UI
- **Configurable Retention**: Adjust retention period via environment variables

## Security Notes

- File uploads are restricted by type and size
- Uploaded files are stored locally in the `uploads/` directory
- Files are automatically cleaned up to prevent disk space issues
- Consider implementing authentication for production use

# Stop the existing RAG ingestion service container

docker-compose stop rag-ingestion-service

# Remove the stopped container (optional but recommended for a clean rebuild)

docker-compose rm -f rag-ingestion-service

# Rebuild and start the service

docker-compose up -d --build rag-ingestion-service

# Optional: check logs to verify it started correctly

docker logs -f <container_name_or_id>

docker logs -f zevo-rag-ingestion
