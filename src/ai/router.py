import google.generativeai as genai
import os
import json
import time

def classify_intent(query: str) -> dict:
    """
    V5.0: Analyzes the user's query and routes it to the appropriate pipeline:
    'sql' (Database), 'rag' (Documents), 'ml' (Predictive), or 'hybrid' (Both).
    Includes automated retry logic with backoff for API rate limits (429).
    """
    status = {
        "success": False, 
        "route": None, 
        "confidence": "low", 
        "reasoning": "", 
        "error": None
    }
    
    prompt = f"""
    You are an intelligent routing agent for an enterprise data system.
    You have access to three distinct analytical tools and data sources:
    1. A PostgreSQL Database ('sql') containing quantitative structured data (sales, revenue, tabular records).
    2. A Document Vector Database ('rag') containing qualitative unstructured text (PDFs, reports, analyses, summaries).
    3. A Predictive Machine Learning Engine ('ml') capable of forecasting future trends and detecting data anomalies.
    
    Determine the best route to answer the user's query.
    - If the query requires calculating numbers, counting records, or querying tabular database information, route to 'sql'.
    - If the query asks for summaries, qualitative explanations, text extraction, or document context, route to 'rag'.
    - If the query asks for future predictions, forecasting, or identifying anomalies/outliers/spikes, route to 'ml'.
    - If the query requires comparing hard database metrics against document claims (e.g., "Does the database revenue match the annual report's claims?"), route to 'hybrid'.
    
    USER QUERY: {query}
    
    Provide your response as a JSON object with strictly these keys:
    "route": string (must be exactly "sql", "rag", "ml", or "hybrid")
    "confidence": string (must be exactly "high", "medium", or "low")
    "reasoning": string (a short explanation of why this route was chosen)
    """
    
    try:
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        
        # Enforce strict JSON output and a very low temperature for deterministic routing
        generation_config = genai.types.GenerationConfig(
            temperature=0.1,
            response_mime_type="application/json"
        )
        
        # Initialize the Gemini 2.5 Flash engine
        model = genai.GenerativeModel('gemini-2.5-flash', generation_config=generation_config)
        
        # Automated Retry Mechanism for 429 Quota Exceeded / Rate Limit errors
        max_retries = 3
        base_delay = 15
        response = None

        for attempt in range(max_retries):
            try:
                response = model.generate_content(prompt)
                break
            except Exception as e:
                error_msg = str(e)
                if any(term in error_msg for term in ["429", "Quota", "ResourceExhausted"]):
                    if attempt < max_retries - 1:
                        time.sleep(base_delay * (attempt + 1))
                        continue
                raise e

        # Clean potential markdown wrapping from response text
        raw_text = response.text.strip()
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:]
        elif raw_text.startswith("```"):
            raw_text = raw_text[3:]
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3]
        raw_text = raw_text.strip()
        
        # Parse the structured JSON output from Gemini
        result_json = json.loads(raw_text)
        
        status["success"] = True
        # Default to 'rag' as a fallback safety if the model hallucinates a route
        status["route"] = result_json.get("route", "rag").lower() 
        status["confidence"] = result_json.get("confidence", "low").lower()
        status["reasoning"] = result_json.get("reasoning", "")
        
    except Exception as e:
        status["error"] = f"Intent classification failed: {str(e)}"
        
    return status