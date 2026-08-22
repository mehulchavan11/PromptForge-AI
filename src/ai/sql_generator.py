import google.generativeai as genai
import os
import re
import time

def generate_sql(prompt: str, table_name: str, schema_info: str) -> dict:
    """
    Converts a natural language prompt into a safe SQL query using Gemini.
    Includes strict prompt guardrails and automated retry logic for API rate limits.
    """
    status = {"success": False, "sql": None, "error": None}
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        status["error"] = "Gemini API key is missing from the environment."
        return status
        
    genai.configure(api_key=api_key)
    
    # 1. System Prompt Engineering (Updated with Strict Guardrails)
    system_instruction = f"""
    You are an expert PostgreSQL data analyst. Convert the natural language prompt into a valid, executable SQL query.
    
    CRITICAL RULE: The target table name is exactly: "{table_name}"
    You MUST use exactly "{table_name}" in your FROM clause. Do not add suffixes, do not invent table names, and do not use aliases that override the base table name in the FROM clause.
    
    Table Schema (Columns & Types):
    {schema_info}
    
    Strict Rules:
    - ONLY generate SELECT queries.
    - NEVER generate queries containing DROP, DELETE, UPDATE, INSERT, ALTER, CREATE, or TRUNCATE.
    - For complex aggregations, use Common Table Expressions (CTEs starting with WITH) or subqueries. NEVER use CREATE TEMP TABLE.
    - Respond strictly with the raw SQL query. Do not include markdown formatting like ```sql.
    - End the query with a semicolon (;).
    """
    
    try:
        model = genai.GenerativeModel("gemini-2.5-flash", system_instruction=system_instruction)
        
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
        
        raw_sql = response.text.strip()
        
        # Clean up any residual markdown formatting
        if raw_sql.startswith("```sql"):
            raw_sql = raw_sql[6:]
        elif raw_sql.startswith("```"):
            raw_sql = raw_sql[3:]
            
        if raw_sql.endswith("```"):
            raw_sql = raw_sql[:-3]
            
        raw_sql = raw_sql.strip()
        
        # 2. The Security Gateway
        if not is_safe_sql(raw_sql):
            status["error"] = "Security Alert: AI generated a forbidden SQL command. Execution blocked."
            return status
            
        status["success"] = True
        status["sql"] = raw_sql
        
    except Exception as e:
        status["error"] = f"AI generation failed: {str(e)}"
        
    return status

def is_safe_sql(sql_query: str) -> bool:
    """
    Validates that the SQL query is strictly a read-only SELECT or WITH statement.
    """
    query_upper = sql_query.upper().lstrip()
    
    # Allow standard SELECTs and CTEs (WITH)
    if not query_upper.startswith("SELECT") and not query_upper.startswith("WITH"):
        return False
        
    forbidden_keywords = [
        "DROP", "DELETE", "UPDATE", "INSERT", "ALTER", 
        "TRUNCATE", "CREATE", "GRANT", "REVOKE"
    ]
    
    for keyword in forbidden_keywords:
        # Uses regex to match exact whole words only
        if re.search(r'\b' + keyword + r'\b', query_upper):
            return False
            
    return True