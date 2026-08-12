import streamlit as st
import pandas as pd
import plotly.express as px
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Import V3 Backend Pipelines from src/
from src.data.ingestion import load_data
from src.data.postgres import save_dataframe_to_postgres, execute_safe_query
from src.ai.sql_generator import generate_sql

# 1. Page & Environment Configuration
st.set_page_config(page_title="Data Hub | PromptForge", page_icon="📊", layout="wide")
load_dotenv()

# 2. Header Section
st.title("📊 Data Intelligence Hub")
st.markdown("""
Upload a dataset to automatically clean, profile, and store it in PostgreSQL. 
Then, create custom visualizations, query it using Natural Language SQL, or generate an AI Executive Report!
""")
st.divider()

# --- FEATURE 1: PIPELINE INGESTION & POSTGRESQL STORAGE ---
uploaded_file = st.file_uploader("Upload structured dataset (CSV, XLSX, JSON)", type=["csv", "xlsx", "xls", "json"])

if uploaded_file is not None:
    try:
        with st.spinner("Processing pipeline & uploading to database..."):
            # Run V3 Ingestion (Load, Error Check, Clean, Whitespace Strip, Deduplicate)
            df, status = load_data(uploaded_file, uploaded_file.name)
            
            if not status["success"]:
                st.error(f"Ingestion Failed: {status['error']}")
                st.stop()
                
            st.success(f"Successfully processed **{uploaded_file.name}**")
            
            # Format clean table name and push to Neon PostgreSQL
            table_name = uploaded_file.name.split('.')[0].lower().replace(" ", "_").replace("-", "_")
            db_status = save_dataframe_to_postgres(df, table_name)
            
            if db_status["success"]:
                st.info(f"💾 Live PostgreSQL Table Created: **`{table_name}`**")
            else:
                st.warning(f"Database warning: {db_status['error']}")

        # --- FEATURE 2: DATA PROFILE KPIs ---
        st.subheader("📋 Data Profile Summary")
        
        total_rows = df.shape[0]
        total_columns = df.shape[1]
        total_missing = df.isnull().sum().sum()
        total_duplicates = df.duplicated().sum()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Rows", f"{total_rows:,}")
        col2.metric("Total Columns", f"{total_columns:,}")
        col3.metric("Missing Values", f"{total_missing:,}", 
                    delta="Action Required" if total_missing > 0 else "Clean", 
                    delta_color="inverse")
        col4.metric("Duplicate Rows", f"{total_duplicates:,}", 
                    delta="Action Required" if total_duplicates > 0 else "Clean", 
                    delta_color="inverse")

        with st.expander("Preview Cleaned Dataset (First 100 Rows)"):
            st.dataframe(df.head(100), use_container_width=True)

        st.divider()

        # --- FEATURE 3: INTERACTIVE CHART BUILDER ---
        st.subheader("📈 Interactive Custom Visualizations")
        
        numeric_columns = df.select_dtypes(include='number').columns.tolist()
        all_columns = df.columns.tolist()

        if len(numeric_columns) > 0:
            chart_col1, chart_col2, chart_col3 = st.columns(3)
            
            with chart_col1:
                chart_type = st.selectbox("Select Chart Type", ["Bar Chart", "Line Chart", "Scatter Plot"])
            with chart_col2:
                x_axis = st.selectbox("Select X-Axis", all_columns)
            with chart_col3:
                y_axis = st.selectbox("Select Y-Axis", numeric_columns)
            
            if chart_type == "Bar Chart":
                fig = px.bar(df, x=x_axis, y=y_axis, title=f"{y_axis} by {x_axis}", template="plotly_white")
            elif chart_type == "Line Chart":
                fig = px.line(df, x=x_axis, y=y_axis, title=f"{y_axis} over {x_axis}", template="plotly_white")
            else:
                fig = px.scatter(df, x=x_axis, y=y_axis, title=f"{y_axis} vs {x_axis}", template="plotly_white")
                
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No numerical columns found to generate custom charts.")

        st.divider()

        # --- FEATURE 4: NATURAL LANGUAGE SQL QUERY ENGINE ---
        st.subheader("🤖 Natural Language Database Query")
        st.caption("Ask questions about your data in plain English. Gemini translates questions into safe SQL executed against PostgreSQL.")
        
        user_question = st.text_input("Ask a question about this data (e.g., 'Show me the top 5 records by revenue')")
        
        if st.button("Generate & Run Query") and user_question:
            with st.spinner("Translating question to SQL..."):
                schema_info = ", ".join([f"{col} ({dtype})" for col, dtype in zip(df.columns, df.dtypes)])
                sql_status = generate_sql(user_question, table_name, schema_info)
                
                if not sql_status["success"]:
                    st.error(sql_status["error"])
                else:
                    st.code(sql_status["sql"], language="sql")
                    
                    try:
                        results_df = execute_safe_query(sql_status["sql"])
                        st.write("### Query Results")
                        st.dataframe(results_df, use_container_width=True)
                    except Exception as e:
                        st.error(f"SQL Execution Error: {str(e)}")

        st.divider()

        # --- FEATURE 5: AI EXECUTIVE BUSINESS REPORT ---
        st.subheader("📝 AI Business Insights Report")
        st.write("Let PromptForge AI analyze overall statistical distributions and generate an executive report.")

        if st.button("Generate Executive Report", type="primary"):
            with st.spinner("Analyzing statistical distributions..."):
                try:
                    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
                    
                    generation_config = {
                        "temperature": 0.2,
                        "top_p": 0.95,
                    }
                    model = genai.GenerativeModel(
                        model_name='gemini-2.5-flash',
                        generation_config=generation_config
                    )

                    data_context = f"""
                    Data Profile:
                    - Total Rows: {df.shape[0]}
                    - Total Columns: {df.shape[1]}

                    Schema and Data Types:
                    {df.dtypes.to_string()}

                    Statistical Summary:
                    {df.describe().to_string()}
                    """

                    prompt = f"""
                    You are an expert Data Analyst and Business Strategist. 
                    Analyze this dataset summary and provide actionable business insights.

                    Dataset Summary:
                    {data_context}

                    Format your response using Markdown:
                    ### 📈 Executive Summary
                    ### 🔍 Key Findings
                    ### ⚠️ Data Quality & Risks
                    ### 💡 Strategic Recommendations
                    """

                    response = model.generate_content(prompt)
                    
                    with st.container(border=True):
                        st.markdown(response.text)

                except Exception as ai_error:
                    st.error(f"AI Generation failed: {str(ai_error)}")

    except Exception as e:
        st.error(f"An error occurred while processing the file: {str(e)}")
else:
    st.info("Please upload a CSV, Excel, or JSON file to begin.")