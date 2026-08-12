from typing import List, Dict, Any

def chunk_document_text(cleaned_text: str, filename: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict[str, Any]]:
    """
    Splits cleaned text into overlapping chunks and attaches standard metadata.
    Uses strict mathematical stepping to guarantee memory safety and prevent infinite loops.
    """
    if not cleaned_text:
        return []
        
    chunks = []
    text_length = len(cleaned_text)
    start = 0
    chunk_id = 1
    
    # Safety Check: Prevent overlap from being larger than or equal to the chunk size
    if overlap >= chunk_size:
        overlap = int(chunk_size * 0.2) # Default to 20% overlap if invalid
        
    # The step dictates exactly how far forward the loop moves every time
    step = chunk_size - overlap
    
    while start < text_length:
        # Define the end of the chunk
        end = min(start + chunk_size, text_length)
        
        chunk_content = cleaned_text[start:end].strip()
        
        if chunk_content:
            chunk_metadata = {
                "chunk_id": chunk_id,
                "source_file": filename,
                "content_length": len(chunk_content),
                "content": chunk_content
            }
            chunks.append(chunk_metadata)
            chunk_id += 1
            
        # Move forward strictly by the mathematical step (guarantees no infinite loops)
        start += step
        
    return chunks