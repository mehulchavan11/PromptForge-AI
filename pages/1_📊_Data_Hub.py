import streamlit as st
import pandas as pd
import io
import os
import plotly.express as px
import google.generativeai as genai
from dotenv import load_dotenv

# 1. Page Configuration (Must be the first Streamlit command)
st.set_page_config(
    page_title="Data Hub | PromptForge",
    page_icon="📊",
    layout="wide"
)

# Load environment variables for Gemini
load_dotenv()

# 2. Header Section
st.title("📊 Data Hub")
st.markdown("""
Upload your dataset to instantly generate data profiles, clean your data, and prepare it for AI-driven insights. 
*Supported formats: CSV, Excel (.xlsx, .xls)*
""")
st.divider()

# 3. File Upload Module
uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    try:
        # Show a loading spinner while processing large files
        with st.spinner("Profiling dataset..."):
            
            # 4. Data Loading Logic
            file_extension = uploaded_file.name.split('.')[-1]
            if file_extension == 'csv':
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"Successfully loaded: **{uploaded_file.name}**")

            # 5. Data Preview Section
            st.subheader("Data Preview")
            # Display the first 100 rows to keep the UI fast
            st.dataframe(df.head(100), use_container_width=True)

            st.divider()

            # 6. Automated EDA (KPI Cards)
            st.subheader("Data Profile Summary")
            
            # Calculate core metrics
            total_rows = df.shape[0]
            total_columns = df.shape[1]
            total_missing = df.isnull().sum().sum()
            total_duplicates = df.duplicated().sum()

            # Display metrics in a clean row
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Rows", f"{total_rows:,}")
            col2.metric("Total Columns", f"{total_columns:,}")
            
            # Highlight missing values and duplicates in red if they exist
            col3.metric("Missing Values", f"{total_missing:,}", 
                        delta="Action Required" if total_missing > 0 else "Clean", 
                        delta_color="inverse")
            col4.metric("Duplicate Rows", f"{total_duplicates:,}", 
                        delta="Action Required" if total_duplicates > 0 else "Clean", 
                        delta_color="inverse")

            st.write("") # Spacer

            # 7. Detailed Schema & Missing Values Breakdown
            schema_col, missing_col = st.columns(2)

            with schema_col:
                st.write("**Column Data Types**")
                # Convert dtypes to string for clean Streamlit rendering
                dtype_df = pd.DataFrame(df.dtypes, columns=['Data Type']).astype(str)
                st.dataframe(dtype_df, use_container_width=True)

            with missing_col:
                st.write("**Missing Values per Column**")
                missing_df = pd.DataFrame(df.isnull().sum(), columns=['Missing Count'])
                # Filter out columns that have 0 missing values
                missing_df = missing_df[missing_df['Missing Count'] > 0]
                
                if missing_df.empty:
                    st.info("🎉 No missing values detected in any column!")
                else:
                    st.dataframe(missing_df, use_container_width=True)

            st.divider()

            # 8. Interactive Data Visualization
            st.subheader("📈 Dynamic Visualizations")
            st.write("Explore your data by selecting variables and chart types below.")

            # Filter columns by data type to prevent charting errors
            # CHANGE IT TO THIS:
            numeric_columns = df.select_dtypes(include='number').columns.tolist()
            all_columns = df.columns.tolist()

            if len(numeric_columns) > 0:
                # Create controls for the user to build their own chart
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
                    
                # Render the chart in Streamlit
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No numerical columns found to generate charts.")

            st.divider()
            
            # 9. AI Insight Generator
            st.subheader("🤖 AI Business Insights")
            st.write("Let PromptForge AI analyze your dataset and generate an executive report.")

            # Use a button so the API isn't called every time the user changes a chart dropdown
            if st.button("Generate Executive Report", type="primary"):
                with st.spinner("Analyzing statistical distributions and generating insights..."):
                    try:
                        # Configure Gemini (Ensure GEMINI_API_KEY is in your .env file)
                        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
                        
                        # Set deterministic parameters for analytical consistency
                        generation_config = {
                            "temperature": 0.2,
                            "top_p": 0.95,
                        }
                        model = genai.GenerativeModel(
                            model_name='gemini-2.5-flash',
                            generation_config=generation_config
                        )

                        # Create a compact data summary context to send to the LLM
                        data_context = f"""
                        Data Profile:
                        - Total Rows: {df.shape[0]}
                        - Total Columns: {df.shape[1]}

                        Schema and Data Types:
                        {df.dtypes.to_string()}

                        Statistical Summary of Numeric Data:
                        {df.describe().to_string()}

                        Sample Data (First 3 rows):
                        {df.head(3).to_string()}
                        """

                        # The core Prompt Engineering for the AI Analyst Persona
                        prompt = f"""
                        You are an expert Data Analyst and Business Strategist. 
                        I am providing you with a statistical summary of a dataset. 
                        Analyze this data and provide actionable business insights.

                        Dataset Summary:
                        {data_context}

                        Format your response exactly as follows using Markdown:
                        
                        ### 📈 Executive Summary
                        [Write a 2-3 sentence overview of the data's primary narrative]

                        ### 🔍 Key Findings
                        [Provide 3-4 bullet points highlighting significant trends, averages, or outliers]

                        ### ⚠️ Data Quality & Risks
                        [Note any missing data issues, concerning patterns, or data quality warnings based on the summary]

                        ### 💡 Strategic Recommendations
                        [Provide 2-3 actionable business next steps based on these findings]
                        """

                        response = model.generate_content(prompt)
                        
                        # Render the AI's markdown response in a clean UI container
                        with st.container(border=True):
                            st.markdown(response.text)

                    except Exception as ai_error:
                        st.error(f"AI Generation failed: {str(ai_error)}")
                        st.info("Make sure your GEMINI_API_KEY is set in your .env file.")

    except Exception as e:
        st.error(f"An error occurred while processing the file: {str(e)}")
else:
    st.info("Please upload a CSV or Excel file to begin.")