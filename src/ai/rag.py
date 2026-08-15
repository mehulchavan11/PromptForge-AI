import google.generativeai as genai
import os
from src.documents.vectorstore import semantic_search

def generate_rag_answer(query: str, top_k: int = 10) -> dict:
    """
    V5.0: Retrieves relevant chunks from ChromaDB (using local embeddings) 
    and forces Gemini to generate an answer using strictly those chunks, with citations.
    """
    status = {"success": False, "answer": None, "sources": [], "error": None}
    
    # 1. Retrieve the Top-K chunks from Vector DB (Now uses rate-limit-free local embeddings)
    search_status = semantic_search(query, n_results=top_k)
    
    if not search_status["success"]:
        status["error"] = search_status["error"]
        return status
        
    chroma_data = search_status["results"]
    
    # If no documents are returned, gracefully exit
    if not chroma_data["documents"] or not chroma_data["documents"][0]:
        status["error"] = "No relevant documents found in the Vector Database to answer this query. Please upload and push chunks first."
        return status
        
    documents = chroma_data["documents"][0]
    metadatas = chroma_data["metadatas"][0]
    
    # 2. Format the retrieved chunks into a strict Context Block
    context_blocks = []
    for doc, meta in zip(documents, metadatas):
        # We pass the metadata directly into the prompt so the AI knows the source name
        source_id = f"{meta['source_file']} | Chunk {meta['chunk_id']}"
        context_blocks.append(f"--- SOURCE: [{source_id}] ---\n{doc}\n")
        
        # Save sources to return to the frontend
        status["sources"].append({"doc": doc, "meta": meta})
        
    full_context = "\n".join(context_blocks)
    
    # 3. Prompt Engineering for RAG and Citations
    prompt = f"""
    You are an expert AI Data Analyst. Your task is to answer the user's question based STRICTLY on the provided context documents below.
    
    CITATION RULES:
    1. You MUST explicitly cite the source for every factual claim you make.
    2. Format your citations inline at the end of the relevant sentence like this: (Source: filename.pdf | Chunk X).
    3. Use the exact Source IDs provided in the context blocks.
    4. If the context documents do not contain the answer, state clearly: "I cannot answer this based on the provided documents." DO NOT invent an answer or use outside knowledge.
    
    CONTEXT DOCUMENTS:
    {full_context}
    
    USER QUESTION: 
    {query}
    """
    
    # 4. Generate the Synthesized Answer
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            status["error"] = "GEMINI_API_KEY is missing from environment variables."
            return status
            
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)
        
        status["success"] = True
        status["answer"] = response.text
        
    except Exception as e:
        status["error"] = f"RAG Generation failed: {str(e)}"
        
    return status