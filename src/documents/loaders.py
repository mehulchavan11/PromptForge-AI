import io
import re
from PyPDF2 import PdfReader
from docx import Document
from typing import Dict, Any

def extract_text_from_file(file_obj, filename: str) -> Dict[str, Any]:
    """
    Extracts text from PDF, DOCX, or TXT files and performs basic cleaning.
    Returns the cleaned text and structural metadata.
    """
    status = {
        "success": False,
        "filename": filename,
        "raw_text": "",
        "cleaned_text": "",
        "error": None,
        "pages_processed": 0
    }
    
    try:
        raw_text = ""
        
        # 1. Multi-Format Ingestion
        if filename.endswith(".pdf"):
            reader = PdfReader(file_obj)
            status["pages_processed"] = len(reader.pages)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    raw_text += page_text + "\n\n"
                    
        elif filename.endswith(".docx"):
            doc = Document(file_obj)
            status["pages_processed"] = 1 # DOCX doesn't have a reliable page count in text extraction
            for para in doc.paragraphs:
                raw_text += para.text + "\n"
                
        elif filename.endswith(".txt"):
            status["pages_processed"] = 1
            # Read bytes and decode
            raw_text = file_obj.read().decode("utf-8")
            
        else:
            status["error"] = f"Unsupported file format: {filename}"
            return status

        # 2. Text Cleaning Engine
        if not raw_text.strip():
            status["error"] = "Extracted text is empty or could not be parsed."
            return status
            
        status["raw_text"] = raw_text
        status["cleaned_text"] = clean_extracted_text(raw_text)
        status["success"] = True
        
    except Exception as e:
        status["error"] = f"Extraction failed: {str(e)}"
        
    return status

def clean_extracted_text(text: str) -> str:
    """
    Normalizes extracted text for downstream chunking and AI processing.
    """
    # Replace multiple spaces with a single space
    cleaned = re.sub(r' +', ' ', text)
    # Replace 3 or more newlines with exactly 2 newlines (paragraph break)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    # Strip leading/trailing whitespace
    return cleaned.strip()