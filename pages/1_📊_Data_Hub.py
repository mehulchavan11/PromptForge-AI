import streamlit as st
import pandas as pd
import plotly.express as px
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Import Core Backend Pipelines
from src.data.ingestion import load_data
from src.data.postgres import save_dataframe_to_postgres, execute_safe_query
from src.ai.sql_generator import generate_sql

# Import V5.0 Enterprise Upgrades
from src.data.cleaner import clean_dataset
from src.data.schema_mapper import detect_schema_roles

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

# --- FEATURE 1: V5.0 PIPELINE INGESTION & POSTGRESQL STORAGE ---
uploaded_file = st.file_uploader("Upload structured dataset (CSV, XLSX, JSON)", type=["csv", "xlsx", "xls", "json"])

if uploaded_file is not None:
    try:
        with st.spinner("Scrubbing data & uploading to database..."):
            # 1. Ingest Raw Dataset
            raw_df, status = load_data(uploaded_file, uploaded_file.name)
            
            if not status["success"]:
                st.error(f"Ingestion Failed: {status['error']}")
                st.stop()
                
            # 2. Apply V5 Data Cleaning Pipeline
            df = clean_dataset(raw_df)
            st.success(f"Successfully processed and cleaned **{uploaded_file.name}**")
            
            # 3. Auto-Detect Semantic Schema
            mapped_schema = detect_schema_roles(df)
            
            # 4. Format clean table name and push to Neon PostgreSQL
            table_name = uploaded_file.name.split('.')[0].lower().replace(" ", "_").replace("-", "_")
            db_status = save_dataframe_to_postgres(df, table_name)
            
            # 5. Store Schema & Table Context into Session Memory for AI Workspace
            st.session_state['auto_schema'] = mapped_schema
            st.session_state['active_table'] = table_name
            
            if db_status["success"]:
                st.info(f"💾 Live PostgreSQL Table Created: **`{table_name}`**")
                with st.expander("🧠 View Auto-Detected Schema Memory"):
                    st.code(mapped_schema, language="text")
            else:
                st.warning(f"Database warning: {db_status['error']}")

        # --- FEATURE 2: DATA PROFILE KPIs ---
        st.divider()
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

        # --- FEATURE 3: INTERACTIVE CHART BUILDER ---
        st.divider()
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

       # --- FEATURE 4: NATURAL LANGUAGE SQL QUERY ENGINE ---
        st.divider()
        st.subheader("🤖 Natural Language Database Query")
        st.caption("Ask questions about your data in plain English. Gemini translates questions into safe SQL executed against PostgreSQL.")
        
        user_question = st.text_input("Ask a question about this data (e.g., 'Show me the top 5 records by revenue')")
        
        if st.button("Generate & Run Query") and user_question:
            with st.spinner("Translating question to SQL..."):
                sql_status = generate_sql(user_question, table_name, mapped_schema)
                
                if not sql_status["success"]:
                    st.error(sql_status["error"])
                else:
                    try:
                        results_df = execute_safe_query(sql_status["sql"])
                        # Save the results into Session Memory so they don't wash out!
                        st.session_state['last_sql_code'] = sql_status["sql"]
                        st.session_state['last_sql_df'] = results_df
                        st.session_state['last_sql_question'] = user_question
                    except Exception as e:
                        st.error(f"SQL Execution Error: {str(e)}")

        # Render the SQL results if they exist in memory for the current question
        if 'last_sql_df' in st.session_state and st.session_state.get('last_sql_question') == user_question:
            st.code(st.session_state['last_sql_code'], language="sql")
            results_df = st.session_state['last_sql_df']
            
            st.write("### Query Results")
            st.dataframe(results_df, use_container_width=True)

            # 🌟 INTERACTIVE SQL CHART BUILDER 🌟
            if len(results_df) > 1 and len(results_df.columns) >= 2:
                all_res_cols = results_df.columns.tolist()
                num_cols = results_df.select_dtypes(include='number').columns.tolist()
                
                st.write("#### 📊 Visualize Query Results")
                sc1, sc2, sc3 = st.columns(3)
                
                with sc1:
                    sql_chart_type = st.selectbox("Chart Type", ["Auto", "Bar Chart", "Line Chart", "Scatter Plot"], key="sql_chart")
                with sc2:
                    sql_x = st.selectbox("X-Axis", all_res_cols, index=0, key="sql_x")
                with sc3:
                    default_y_idx = all_res_cols.index(num_cols[0]) if num_cols else 0
                    sql_y = st.selectbox("Y-Axis", all_res_cols, index=default_y_idx, key="sql_y")
                
                # Render the chosen chart
                if sql_chart_type == "Bar Chart" or (sql_chart_type == "Auto" and len(num_cols) > 0 and len(all_res_cols) > len(num_cols)):
                    fig = px.bar(results_df.sort_values(by=sql_y, ascending=False), x=sql_x, y=sql_y, template="plotly_white")
                    st.plotly_chart(fig, use_container_width=True)
                elif sql_chart_type == "Line Chart" or (sql_chart_type == "Auto" and len(num_cols) >= 2):
                    fig = px.line(results_df, x=sql_x, y=sql_y, template="plotly_white", markers=True)
                    st.plotly_chart(fig, use_container_width=True)
                elif sql_chart_type == "Scatter Plot":
                    fig = px.scatter(results_df, x=sql_x, y=sql_y, template="plotly_white")
                    st.plotly_chart(fig, use_container_width=True)

        # --- FEATURE 5: AI EXECUTIVE BUSINESS REPORT ---
        st.divider()
        st.subheader("📝 AI Business Insights Report")
        st.write("Let PromptForge AI analyze overall statistical distributions and generate an executive report.")

        if st.button("Generate Executive Report", type="primary"):
            with st.spinner("Analyzing statistical distributions..."):
                try:
                    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
                    model = genai.GenerativeModel(model_name='gemini-2.5-flash')

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
                    Dataset Summary: {data_context}
                    
                    Format your response using Markdown:
                    ### 📈 Executive Summary
                    ### 🔍 Key Findings
                    ### ⚠️ Data Quality & Risks
                    ### 💡 Strategic Recommendations
                    """

                    response = model.generate_content(prompt)
                    
                    # Save report to memory so it doesn't wash out!
                    st.session_state['exec_report'] = response.text

                except Exception as ai_error:
                    st.error(f"AI Generation failed: {str(ai_error)}")

        # Render the report if it exists in memory
        if 'exec_report' in st.session_state:
            with st.container(border=True):
                st.markdown(st.session_state['exec_report'])

    except Exception as e:
        st.error(f"An error occurred while processing the file: {str(e)}")
else:
    st.info("Please upload a CSV, Excel, or JSON file to begin.")