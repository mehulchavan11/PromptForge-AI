import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv
import PyPDF2
import io

# 1. Page Configuration
st.set_page_config(
    page_title="Knowledge Hub | PromptForge",
    page_icon="📚",
    layout="wide"
)

load_dotenv()

# Configure Gemini
try:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    generation_config = {"temperature": 0.3, "top_p": 0.9}
    model = genai.GenerativeModel(model_name='gemini-2.5-flash', generation_config=generation_config)
except Exception as e:
    st.error("Failed to configure Gemini API. Please check your .env file.")

# 2. Header Section
st.title("📚 Knowledge Hub")
st.markdown("Upload documents to instantly extract intelligence, summarize long-form text, and isolate key entities.")
st.divider()

# Helper function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text

# 3. Document Upload Section
uploaded_file = st.file_uploader("Upload a document for analysis", type=["txt", "pdf"])

if uploaded_file is not None:
    # 4. Text Extraction
    with st.spinner("Reading document..."):
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension == 'pdf':
            document_text = extract_text_from_pdf(uploaded_file)
        else:
            document_text = uploaded_file.getvalue().decode("utf-8")
            
        st.success(f"Successfully loaded: **{uploaded_file.name}** ({len(document_text)} characters)")

    st.divider()

    # 5. The Unified Workspace (Using Tabs)
    tab1, tab2, tab3 = st.tabs(["📝 Document Summarizer", "🔍 Insight & Entity Extractor", "📄 View Raw Text"])

    # --- TAB 1: SUMMARIZER ---
    with tab1:
        st.subheader("Generate a Structured Summary")
        summary_format = st.radio("Select Output Format:", ["Executive Summary", "Bullet Points", "Key Action Items"], horizontal=True)
        
        if st.button("Generate Summary", type="primary"):
            with st.spinner("Summarizing document..."):
                prompt = f"""
                You are an expert technical reader and summarizer.
                Analyze the following document and provide a summary in the format of: {summary_format}.
                Keep the output professional, concise, and focused on the most critical information.

                Document Text:
                {document_text[:15000]}  # Slicing to ensure we don't blow up token limits unexpectedly
                """
                
                try:
                    response = model.generate_content(prompt)
                    with st.container(border=True):
                        st.markdown(response.text)
                except Exception as e:
                    st.error(f"Summarization failed: {e}")

    # --- TAB 2: INSIGHT EXTRACTOR ---
    with tab2:
        st.subheader("Extract Specific Entities or Insights")
        extraction_target = st.text_input("What would you like to extract? (e.g., 'Names and Dates', 'Financial Metrics', 'Software Tools')")
        
        if st.button("Extract Insights", type="primary"):
            if extraction_target:
                with st.spinner(f"Extracting {extraction_target}..."):
                    prompt = f"""
                    You are an elite data extraction AI.
                    Review the following document and extract the following specific information: {extraction_target}.
                    Format your response strictly as a clean Markdown table. If the information is not present, state that clearly.

                    Document Text:
                    {document_text[:15000]}
                    """
                    
                    try:
                        response = model.generate_content(prompt)
                        with st.container(border=True):
                            st.markdown(response.text)
                    except Exception as e:
                        st.error(f"Extraction failed: {e}")
            else:
                st.warning("Please specify what you want to extract.")

    # --- TAB 3: RAW TEXT PREVIEW ---
    with tab3:
        st.subheader("Extracted Text Preview")
        # Displaying in a scrolling text area to save space
        st.text_area("Document Content", document_text, height=300, disabled=True)

else:
    st.info("Please upload a PDF or TXT file to begin knowledge extraction.")