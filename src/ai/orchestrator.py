import google.generativeai as genai
import os
import pandas as pd
from src.ai.router import classify_intent
from src.ai.sql_generator import generate_sql
from src.data.postgres import execute_safe_query
from src.ai.rag import generate_rag_answer

# V5.0 Imports: Machine Learning Engine
from src.ai.ml_engine import run_anomaly_detection, run_time_series_forecast

def process_user_query(query: str, table_name: str = None, schema_info: str = "", df: pd.DataFrame = None) -> dict:
    """
    V5.0 Orchestrator: Routes and executes queries across SQL, RAG, ML, or Hybrid pipelines.
    """
    output = {
        "success": False,
        "route": "rag",
        "reasoning": "",
        "sql_result": None,
        "rag_result": None,
        "ml_result": None, # Added for V5.0
        "final_synthesis": "",
        "error": None
    }
    
    # 1. Route the intent using our V5.0 AI Router
    route_status = classify_intent(query)
    
    if not route_status["success"]:
        output["error"] = route_status["error"]
        return output
        
    output["route"] = route_status["route"]
    output["reasoning"] = route_status["reasoning"]
    
    try:
        # --- PATH A: SQL ONLY ---
        if output["route"] == "sql":
            if not table_name:
                output["error"] = "SQL routing selected, but no active database table was provided."
                return output
                
            sql_status = generate_sql(query, table_name, schema_info)
            if not sql_status["success"]:
                output["error"] = sql_status["error"]
                return output
                
            db_df = execute_safe_query(sql_status["sql"])
            output["sql_result"] = {
                "sql": sql_status["sql"],
                "data": db_df
            }
            output["final_synthesis"] = f"Generated and executed SQL query against table `{table_name}`."
            output["success"] = True

        # --- PATH B: RAG ONLY ---
        elif output["route"] == "rag":
            rag_status = generate_rag_answer(query, top_k=25)
            if not rag_status["success"]:
                output["error"] = rag_status["error"]
                return output
                
            output["rag_result"] = rag_status
            output["final_synthesis"] = rag_status["answer"]
            output["success"] = True

        # --- PATH C: PREDICTIVE ML PIPELINE (V5.0 FEATURE) 🔮 ---
        elif output["route"] == "ml":
            if df is None and not table_name:
                output["error"] = "ML routing selected, but no dataset was provided."
                return output
                
            # If df isn't passed directly from Streamlit memory, fetch it from Postgres
            ml_df = df if df is not None else execute_safe_query(f"SELECT * FROM {table_name}")
            
            query_lower = query.lower()
            
            # --- ENTERPRISE FIX: DYNAMIC COLUMN MAPPING ---
            # 1. Find the target metric dynamically (money, revenue, or first numeric)
            numeric_columns = ml_df.select_dtypes(include=["number"]).columns.tolist()
            dynamic_target = 'money' if 'money' in ml_df.columns else ('revenue' if 'revenue' in ml_df.columns else numeric_columns[0])
            
            # 2. Find the time column dynamically (hour, date, time)
            time_cols = [c for c in ml_df.columns if 'time' in c or 'date' in c or 'hour' in c]
            dynamic_date = time_cols[0] if time_cols else ml_df.columns[0]
            # ----------------------------------------------
            
            # Simple heuristic to trigger the correct ML function
            if "predict" in query_lower or "forecast" in query_lower or "trend" in query_lower:
                ml_status = run_time_series_forecast(ml_df, date_col=dynamic_date, target_col=dynamic_target)
            else:
                ml_status = run_anomaly_detection(ml_df, target_col=dynamic_target)
                
            if not ml_status.get("success"):
                output["error"] = ml_status.get("error", "Unknown ML Engine failure.")
                return output
                
            output["ml_result"] = ml_status
            
            # Synthesize the raw ML output into a readable narrative
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            model = genai.GenerativeModel('gemini-2.5-flash')
            
            ml_prompt = f"""
            You are a Lead Data Scientist. 
            The user asked a predictive question: "{query}"
            
            Here is the raw output from our Machine Learning Engine:
            {ml_status}
            
            Write a clear, executive-friendly summary of these ML results. 
            Highlight key predictions or anomalies, and explain what they mean for the business.
            """
            
            ml_response = model.generate_content(ml_prompt)
            output["final_synthesis"] = ml_response.text
            output["success"] = True

        # --- PATH D: HYBRID PIPELINE (SIGNATURE V4 FEATURE) ⭐ ---
        elif output["route"] == "hybrid":
            # 1. Execute SQL Pipeline if table exists
            if table_name:
                sql_status = generate_sql(query, table_name, schema_info)
                if sql_status["success"]:
                    try:
                        db_df = execute_safe_query(sql_status["sql"])
                        output["sql_result"] = {"sql": sql_status["sql"], "data": db_df}
                    except Exception as e:
                        output["sql_result"] = {"error": str(e)}
            
            # 2. Execute RAG Pipeline
            rag_status = generate_rag_answer(query, top_k=25)
            if rag_status["success"]:
                output["rag_result"] = rag_status

            # 3. Reconcile both sources using Gemini Synthesis
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            model = genai.GenerativeModel('gemini-2.5-flash')
            
            sql_context = output["sql_result"]["data"].to_string() if output.get("sql_result") and "data" in output["sql_result"] else "No SQL data available."
            rag_context = output["rag_result"]["answer"] if output.get("rag_result") else "No document data available."
            
            synthesis_prompt = f"""
            You are a Lead Business Intelligence Analyst.
            The user asked a hybrid question requiring data from both a Structured SQL Database and Unstructured Document Reports.
            
            USER QUESTION: {query}
            
            STRUCTURED DATABASE RESULT (PostgreSQL):
            {sql_context}
            
            UNSTRUCTURED DOCUMENT RESULT (ChromaDB RAG):
            {rag_context}
            
            TASK:
            1. Synthesize a unified, comprehensive answer comparing both sources.
            2. State clearly whether the SQL database findings align with or contradict the document claims.
            3. Highlight key metrics from the database and keep the document inline citations intact.
            """
            
            synthesis_response = model.generate_content(synthesis_prompt)
            output["final_synthesis"] = synthesis_response.text
            output["success"] = True

    except Exception as e:
        output["error"] = f"Orchestration failure: {str(e)}"

    return output