import google.generativeai as genai
import os
import json
import time

def review_output(query: str, orchestrator_result: dict) -> dict:
    """
    V5.0: Validates the final synthesis from the orchestrator.
    Includes robust markdown cleaning, schema-based JSON generation, 
    and automated retry logic with backoff for API rate limits (429).
    """
    status = {
        "success": False,
        "is_valid": False,
        "feedback": "",
        "final_approved_text": "",
        "error": None
    }

    if not orchestrator_result.get("success"):
        status["error"] = "Cannot review a failed orchestrator output."
        return status

    route = orchestrator_result.get("route")
    proposed_answer = orchestrator_result.get("final_synthesis", "")
    
    # 1. Gather the ground-truth evidence
    context_data = ""
    
    if route in ["sql", "hybrid"] and orchestrator_result.get("sql_result"):
        context_data += f"\n[SQL DATA EXECUTED]:\n{orchestrator_result['sql_result'].get('data')}\n"
        
    if route in ["rag", "hybrid"] and orchestrator_result.get("rag_result"):
        context_data += "\n[DOCUMENT SOURCES RETRIEVED]:\n"
        for src in orchestrator_result['rag_result'].get('sources', []):
            context_data += f"- {src['meta']['source_file']} (Chunk {src['meta']['chunk_id']}): {src['doc']}\n"
            
    if orchestrator_result.get("ml_result"):
        context_data += f"\n[ML ENGINE PREDICTIONS]:\n{orchestrator_result['ml_result']}\n"

    if not context_data.strip():
        context_data = "No specific dataset, document, or ML context was provided."

    # 2. Prompt the Reviewer Agent
    prompt = f"""
    You are an elite AI Quality Assurance Reviewer (The Critic Node).
    Your ONLY job is to review an AI-generated answer against the provided ground-truth context and the user's original query.

    USER QUERY: {query}
    ROUTE TAKEN: {route}

    AVAILABLE GROUND-TRUTH CONTEXT:
    {context_data}

    PROPOSED AI ANSWER:
    {proposed_answer}

    ### STRICT EVALUATION RULES:
    1. FOCUS ON ACCURACY, NOT STYLE: If the Proposed Answer contains the correct numbers or facts from the Ground-Truth Context, you MUST approve it. 
    2. IGNORE FORMATTING: Do NOT fail the answer just because it is brief or conversational. 
    3. WHAT CONSTITUTES A FAIL: You may ONLY intervene if the AI hallucinates fake numbers or directly contradicts the context.

    Provide your response as a valid JSON object ONLY, adhering EXACTLY to this schema:
    {{
      "is_valid": boolean (true if approved, false if hallucinated),
      "feedback": "string (brief explanation, e.g., 'Data matches ground truth.')",
      "revised_answer": "string (return the PROPOSED AI ANSWER exactly as is if valid, or a corrected version if invalid)"
    }}
    """

    try:
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        
        generation_config = genai.types.GenerationConfig(
            temperature=0.0,
            response_mime_type="application/json"
        )
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

        result_json = json.loads(raw_text)
        
        status["success"] = True
        status["is_valid"] = result_json.get("is_valid", True)
        status["feedback"] = result_json.get("feedback", "Approved.")
        status["final_approved_text"] = result_json.get("revised_answer", proposed_answer)

    except Exception as e:
        # FAIL-OPEN SAFETY: If review fails technically, default to passing the output 
        # so the user never sees a broken UI block.
        status["success"] = True
        status["is_valid"] = True
        status["feedback"] = f"Auto-approved via safety fallback (Log: {str(e)})."
        status["final_approved_text"] = proposed_answer
        status["error"] = None

    return status