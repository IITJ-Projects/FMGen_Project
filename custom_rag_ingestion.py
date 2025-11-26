#!/usr/bin/env python3
"""
Custom RAG Ingestion Script
Allows you to ingest your own documents into the RAG service
"""

import requests
import json
import os
from typing import List, Dict, Any

# RAG service endpoint
RAG_URL = "http://localhost:8004"

def ingest_documents(documents: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Ingest documents into RAG service"""
    
    payload = {
        "documents": documents
    }
    
    try:
        response = requests.post(
            f"{RAG_URL}/ingest",
            headers={"Content-Type": "application/json"},
            json=payload
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"Failed to ingest documents: {e}")
        return None

def ingest_from_file(file_path: str) -> Dict[str, Any]:
    """Ingest documents from a JSON file"""
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            documents = data
        elif isinstance(data, dict) and 'documents' in data:
            documents = data['documents']
        else:
            print("Invalid file format. Expected list of documents or {'documents': [...]}")
            return None
        
        return ingest_documents(documents)
        
    except Exception as e:
        print(f"Failed to read file {file_path}: {e}")
        return None

def ingest_from_text(text: str, doc_id: str = None, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
    """Ingest a single text document"""
    
    if doc_id is None:
        import uuid
        doc_id = f"doc_{uuid.uuid4().hex[:8]}"
    
    if metadata is None:
        metadata = {}
    
    document = {
        "id": doc_id,
        "text": text,
        "metadata": metadata
    }
    
    return ingest_documents([document])

def test_retrieval(query: str, top_k: int = 5) -> Dict[str, Any]:
    """Test retrieval from RAG service"""
    
    payload = {
        "query": query,
        "top_k": top_k
    }
    
    try:
        response = requests.post(
            f"{RAG_URL}/retrieve",
            headers={"Content-Type": "application/json"},
            json=payload
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"Failed to retrieve documents: {e}")
        return None

def main():
    """Main function with examples"""
    
    print("=== Custom RAG Ingestion Script ===\n")
    
    # Example 1: Ingest single document
    print("1. Ingesting single document...")
    result = ingest_from_text(
        "Deep Learning is a subset of machine learning that uses neural networks with multiple layers to model complex patterns in data.",
        doc_id="deep_learning_001",
        metadata={"category": "technology", "topic": "deep learning"}
    )
    print(f"Result: {result}\n")
    
    # Example 2: Ingest multiple documents
    print("2. Ingesting multiple documents...")
    documents = [
        {
            "id": "nlp_001",
            "text": "Natural Language Processing (NLP) is a field of AI that focuses on the interaction between computers and human language.",
            "metadata": {"category": "technology", "topic": "nlp"}
        },
        {
            "id": "computer_vision_001", 
            "text": "Computer Vision is a field of AI that enables machines to interpret and understand visual information from the world.",
            "metadata": {"category": "technology", "topic": "computer vision"}
        }
    ]
    
    result = ingest_documents(documents)
    print(f"Result: {result}\n")
    
    # Example 3: Test retrieval
    print("3. Testing retrieval...")
    result = test_retrieval("What is deep learning?", top_k=3)
    if result:
        print(f"Found {result['total_found']} documents")
        for i, doc in enumerate(result['documents'], 1):
            print(f"  {i}. Score: {doc.get('score', 'N/A'):.3f}")
            print(f"     Text: {doc.get('text', 'N/A')[:100]}...")
    print()

if __name__ == "__main__":
    main()
