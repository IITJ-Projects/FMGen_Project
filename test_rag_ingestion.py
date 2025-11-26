#!/usr/bin/env python3
"""
Test script to add sample documents to RAG service
"""

import requests
import json

# RAG service endpoint
RAG_URL = "http://localhost:8004"

# Sample documents about AI and technology
sample_documents = [
    {
        "id": "doc_001",
        "text": "Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines capable of performing tasks that typically require human intelligence. These tasks include learning, reasoning, problem-solving, perception, and language understanding.",
        "metadata": {
            "category": "technology",
            "topic": "artificial intelligence",
            "source": "sample_data"
        }
    },
    {
        "id": "doc_002", 
        "text": "Machine Learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed. It uses algorithms to identify patterns in data and make predictions or decisions.",
        "metadata": {
            "category": "technology",
            "topic": "machine learning",
            "source": "sample_data"
        }
    },
    {
        "id": "doc_003",
        "text": "Natural Language Processing (NLP) is a field of AI that focuses on the interaction between computers and human language. It enables machines to understand, interpret, and generate human language in a meaningful way.",
        "metadata": {
            "category": "technology", 
            "topic": "natural language processing",
            "source": "sample_data"
        }
    },
    {
        "id": "doc_004",
        "text": "Deep Learning is a subset of machine learning that uses neural networks with multiple layers to model and understand complex patterns in data. It has been particularly successful in image recognition, speech recognition, and natural language processing.",
        "metadata": {
            "category": "technology",
            "topic": "deep learning", 
            "source": "sample_data"
        }
    },
    {
        "id": "doc_005",
        "text": "Large Language Models (LLMs) are AI models trained on vast amounts of text data to understand and generate human language. Examples include GPT, LLaMA, and BERT, which can perform tasks like text generation, translation, and question answering.",
        "metadata": {
            "category": "technology",
            "topic": "large language models",
            "source": "sample_data"
        }
    }
]

def test_rag_ingestion():
    """Test ingesting documents into RAG service"""
    
    print("Testing RAG service ingestion...")
    
    # Test health check first
    try:
        health_response = requests.get(f"{RAG_URL}/health")
        print(f"RAG Health Status: {health_response.status_code}")
        if health_response.status_code == 200:
            print(f"Health Response: {health_response.json()}")
    except Exception as e:
        print(f"Health check failed: {e}")
        return
    
    # Test document ingestion
    try:
        ingestion_data = {
            "documents": sample_documents
        }
        
        response = requests.post(
            f"{RAG_URL}/ingest",
            headers={"Content-Type": "application/json"},
            json=ingestion_data
        )
        
        print(f"Ingestion Response Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Ingested {result['ingested_count']} documents")
            print(f"Failed: {result['failed_count']} documents")
            print(f"Processing time: {result['processing_time']:.3f}s")
        else:
            print(f"Ingestion failed: {response.text}")
            
    except Exception as e:
        print(f"Ingestion test failed: {e}")

def test_rag_retrieval():
    """Test retrieving documents from RAG service"""
    
    print("\nTesting RAG service retrieval...")
    
    # Test queries
    test_queries = [
        "artificial intelligence",
        "machine learning", 
        "natural language processing",
        "deep learning",
        "large language models"
    ]
    
    for query in test_queries:
        try:
            retrieval_data = {
                "query": query,
                "top_k": 3
            }
            
            response = requests.post(
                f"{RAG_URL}/retrieve",
                headers={"Content-Type": "application/json"},
                json=retrieval_data
            )
            
            print(f"\nQuery: '{query}'")
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"Found {result['total_found']} documents")
                print(f"Processing time: {result['processing_time']:.3f}s")
                
                for i, doc in enumerate(result['documents'], 1):
                    print(f"  {i}. Score: {doc.get('score', 'N/A'):.3f}")
                    print(f"     Text: {doc.get('text', 'N/A')[:100]}...")
            else:
                print(f"Retrieval failed: {response.text}")
                
        except Exception as e:
            print(f"Retrieval test failed for '{query}': {e}")

if __name__ == "__main__":
    test_rag_ingestion()
    test_rag_retrieval()
