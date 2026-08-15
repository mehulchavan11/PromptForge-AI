import google.generativeai as genai
import os
import json

def review_output(query: str, orchestrator_result: dict) -> dict:
    """
    V5.0: Validates the final synthesis from the orchestrator.
    Checks for hallucinations, ensures citations exist, and verifies SQL/ML alignment.
    Outputs structured JSON.
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
    
    # 1. Gather the ground-truth evidence that the Orchestrator used
    context_data = ""
    
    # --- SQL Context ---
    if route in ["sql", "hybrid"] and orchestrator_result.get("sql_result"):
        context_data += f"\n[SQL DATA EXECUTED]:\n{orchestrator_result['sql_result'].get('data')}\n"
        
    # --- RAG Context ---
    if route in ["rag", "hybrid"] and orchestrator_result.get("rag_result"):
        context_data += "\n[DOCUMENT SOURCES RETRIEVED]:\n"
        for src in orchestrator_result['rag_result'].get('sources', []):
            context_data += f"- {src['meta']['source_file']} (Chunk {src['meta']['chunk_id']}): {src['doc']}\n"
            
    # --- ML Context (V5.0 Addition) ---
    if orchestrator_result.get("ml_result"):
        context_data += f"\n[ML ENGINE PREDICTIONS]:\n{orchestrator_result['ml_result']}\n"

    # Safety fallback
    if not context_data.strip():
        context_data = "No specific dataset, document, or ML context was provided."

    # 2. Prompt the Reviewer Agent
    prompt = f"""
    You are an elite AI Quality Assurance Reviewer.
    Your job is to review an AI-generated answer against the provided ground-truth context and the user's original query.

    USER QUERY: {query}
    ROUTE TAKEN: {route}

    AVAILABLE GROUND-TRUTH CONTEXT:
    {context_data}

    PROPOSED AI ANSWER:
    {proposed_answer}

    VALIDATION RULES:
    1. SQL Routes: Does the answer accurately reflect the provided SQL data without making up numbers?
    2. RAG Routes: Are there inline citations? Does the answer rely ONLY on the provided document sources?
    3. ML Routes: Does the answer accurately reflect the provided ML Engine predictions without fabricating numbers?
    4. Hybrid Routes: Does the answer accurately synthesize the sources without hallucinating?

    Evaluate the PROPOSED AI ANSWER.
    Provide your response as a JSON object with strictly these keys:
    "is_valid": boolean (true if the answer passes all rules, false if it hallucinates or fails rules)
    "feedback": string (brief explanation of why it passed or failed, e.g., 'Accurately cited sources.')
    "revised_answer": string (If is_valid is true, return the exact PROPOSED AI ANSWER. If false, rewrite the answer to be safe and accurate based ONLY on the context).
    """

    try:
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        
        # Enforce strict JSON output
        generation_config = genai.types.GenerationConfig(
            temperature=0.1,
            response_mime_type="application/json"
        )
        model = genai.GenerativeModel('gemini-2.5-flash', generation_config=generation_config)
        
        response = model.generate_content(prompt)
        result_json = json.loads(response.text)
        
        status["success"] = True
        status["is_valid"] = result_json.get("is_valid", False)
        status["feedback"] = result_json.get("feedback", "No feedback provided.")
        status["final_approved_text"] = result_json.get("revised_answer", proposed_answer)

    except Exception as e:
        status["error"] = f"Reviewer validation failed: {str(e)}"

    return status