#!/usr/bin/env python3
"""
Test script for RAG Ingestion Application
Demonstrates file upload, processing, and retrieval
"""

import requests
import json
import time
from pathlib import Path

# RAG Ingestion App endpoint
RAG_INGESTION_URL = "http://localhost:8005"

def test_health():
    """Test health endpoint"""
    print("=== Testing Health Endpoint ===")
    try:
        response = requests.get(f"{RAG_INGESTION_URL}/health")
        if response.status_code == 200:
            health = response.json()
            print(f"Status: {health['status']}")
            print(f"RAG Service: {health['rag_service']}")
            print(f"LLM Service: {health['llm_service']}")
            return True
        else:
            print(f"Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"Health check error: {e}")
        return False

def create_sample_text_file():
    """Create a sample text file for testing"""
    sample_text = """
    Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines capable of performing tasks that typically require human intelligence. These tasks include learning, reasoning, problem-solving, perception, and language understanding.

    Machine Learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed. It uses algorithms to identify patterns in data and make predictions or decisions based on those patterns.

    Deep Learning is a subset of machine learning that uses neural networks with multiple layers to model and understand complex patterns in data. It has been particularly successful in image recognition, speech recognition, and natural language processing.

    Natural Language Processing (NLP) is a field of AI that focuses on the interaction between computers and human language. It enables machines to understand, interpret, and generate human language in a meaningful way.

    Computer Vision is a field of AI that enables machines to interpret and understand visual information from the world. It involves developing algorithms and systems that can process, analyze, and make decisions based on visual data.
    """
    
    file_path = Path("sample_ai_document.txt")
    with open(file_path, 'w') as f:
        f.write(sample_text)
    
    return file_path

def test_file_upload(file_path):
    """Test file upload"""
    print(f"\n=== Testing File Upload: {file_path.name} ===")
    
    try:
        with open(file_path, 'rb') as f:
            files = {'file': (file_path.name, f, 'text/plain')}
            response = requests.post(f"{RAG_INGESTION_URL}/upload", files=files)
        
        if response.status_code == 200:
            result = response.json()
            print(f"Upload successful!")
            print(f"File ID: {result['file_id']}")
            print(f"Filename: {result['filename']}")
            print(f"Status: {result['status']}")
            return result['file_id']
        else:
            print(f"Upload failed: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"Upload error: {e}")
        return None

def test_file_processing(file_id):
    """Test file processing"""
    print(f"\n=== Testing File Processing: {file_id} ===")
    
    try:
        response = requests.post(f"{RAG_INGESTION_URL}/process/{file_id}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Processing successful!")
            print(f"Chunks created: {result['chunks_created']}")
            print(f"Embeddings generated: {result['embeddings_generated']}")
            print(f"Processing time: {result['processing_time']:.2f}s")
            print(f"Status: {result['status']}")
            return True
        else:
            print(f"Processing failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"Processing error: {e}")
        return False

def test_retrieval():
    """Test retrieval from RAG service"""
    print(f"\n=== Testing Retrieval ===")
    
    # Test queries
    test_queries = [
        "What is artificial intelligence?",
        "How does machine learning work?",
        "What is deep learning?",
        "Explain natural language processing"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        try:
            response = requests.post(
                "http://localhost:8004/retrieve",
                json={"query": query, "top_k": 3}
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"Found {result['total_found']} documents")
                for i, doc in enumerate(result['documents'], 1):
                    print(f"  {i}. Score: {doc.get('score', 'N/A'):.3f}")
                    print(f"     Text: {doc.get('text', 'N/A')[:100]}...")
            else:
                print(f"Retrieval failed: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"Retrieval error: {e}")

def test_list_files():
    """Test listing uploaded files"""
    print(f"\n=== Testing File Listing ===")
    
    try:
        response = requests.get(f"{RAG_INGESTION_URL}/files")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Found {len(result['files'])} files:")
            for file_info in result['files']:
                print(f"  - {file_info['filename']} ({file_info['size']} bytes)")
        else:
            print(f"File listing failed: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"File listing error: {e}")

def main():
    """Main test function"""
    print("=== RAG Ingestion Application Test ===\n")
    
    # Test health
    if not test_health():
        print("Health check failed. Make sure the RAG ingestion app is running.")
        return
    
    # Create sample file
    sample_file = create_sample_text_file()
    print(f"Created sample file: {sample_file}")
    
    # Test file upload
    file_id = test_file_upload(sample_file)
    if not file_id:
        print("File upload failed. Stopping test.")
        return
    
    # Test file processing
    if not test_file_processing(file_id):
        print("File processing failed. Stopping test.")
        return
    
    # Wait a moment for processing to complete
    time.sleep(2)
    
    # Test retrieval
    test_retrieval()
    
    # Test file listing
    test_list_files()
    
    print(f"\n=== Test Completed ===")
    print(f"Sample file created: {sample_file}")
    print(f"File ID: {file_id}")

if __name__ == "__main__":
    main()
