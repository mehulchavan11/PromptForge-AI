import chromadb
from chromadb.utils import embedding_functions
import os
from typing import List, Dict, Any

def get_chroma_collection(collection_name: str = "promptforge_knowledge"):
    """Initializes a persistent local ChromaDB and returns the collection using local embeddings."""
    db_path = os.path.join(os.getcwd(), "chroma_db")
    client = chromadb.PersistentClient(path=db_path)
    
    # 🌟 ENTERPRISE UPGRADE: Use Local HuggingFace Embeddings (No API limits!)
    local_ef = embedding_functions.DefaultEmbeddingFunction()
    
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=local_ef
    )
    return collection

def store_chunks_in_chroma(chunks: List[Dict[str, Any]]) -> dict:
    """Takes text chunks and metadata, and pushes them to ChromaDB."""
    status = {"success": False, "error": None, "chunks_inserted": 0}
    
    if not chunks:
        status["error"] = "No chunks provided to store."
        return status
        
    try:
        # Connect to DB using the updated local embeddings
        collection = get_chroma_collection()
        
        documents = []
        metadatas = []
        ids = []
        
        for chunk in chunks:
            documents.append(chunk["content"])
            metadata = {
                "source_file": chunk["source_file"],
                "chunk_id": chunk["chunk_id"],
                "content_length": chunk["content_length"]
            }
            metadatas.append(metadata)
            unique_id = f"{chunk['source_file']}_chunk_{chunk['chunk_id']}"
            ids.append(unique_id)
            
        # Add to the database (Using your original 'upsert' to prevent duplicate crashes)
        collection.upsert(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        status["success"] = True
        status["chunks_inserted"] = len(chunks)
        
    except Exception as e:
        status["error"] = f"Failed to store in Vector DB: {str(e)}"
        
    return status

def semantic_search(query: str, n_results: int = 3) -> dict:
    """Searches the Vector DB for chunks most semantically similar to the query."""
    status = {"success": False, "results": None, "error": None}
    try:
        # Connect to DB using the updated local embeddings
        collection = get_chroma_collection()
        
        results = collection.query(query_texts=[query], n_results=n_results)
        status["success"] = True
        status["results"] = results
    except Exception as e:
        status["error"] = f"Semantic search failed: {str(e)}"
    return status