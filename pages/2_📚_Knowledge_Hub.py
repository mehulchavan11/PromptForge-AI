import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Import V3 Backend Document Pipelines from src/
from src.documents.loaders import extract_text_from_file
from src.documents.chunking import chunk_document_text

# 1. Page Configuration
st.set_page_config(
    page_title="Knowledge Hub | PromptForge",
    page_icon="📚",
    layout="wide"
)

load_dotenv()

# Configure Gemini API
try:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    generation_config = {"temperature": 0.3, "top_p": 0.9}
    model = genai.GenerativeModel(model_name='gemini-2.5-flash', generation_config=generation_config)
except Exception as e:
    st.error("Failed to configure Gemini API. Please verify your GEMINI_API_KEY in the .env file.")

# 2. Header Section
st.title("📚 Knowledge Intelligence Hub")
st.markdown("""
Upload unstructured documents (PDF, DOCX, TXT) to clean, extract, chunk, and attach structural metadata—preparing 
your knowledge base for AI-driven summarization, entity extraction, and Version 4.0 RAG systems.
""")
st.divider()

# 3. Document Ingestion Section
uploaded_doc = st.file_uploader("Upload a document for analysis", type=["pdf", "docx", "txt"])

if uploaded_doc is not None:
    # Sidebar Chunking Controls
    st.sidebar.header("⚙️ Chunking Engine Settings")
    chunk_size = st.sidebar.slider("Chunk Size (characters)", min_value=200, max_value=2000, value=1000, step=100)
    chunk_overlap = st.sidebar.slider("Chunk Overlap (characters)", min_value=0, max_value=500, value=200, step=50)

    with st.spinner("Executing document pipeline..."):
        # Run V3 Ingestion Engine (PDF, DOCX, TXT extraction & cleaning)
        doc_status = extract_text_from_file(uploaded_doc, uploaded_doc.name)
        
        if not doc_status["success"]:
            st.error(f"Extraction Error: {doc_status['error']}")
            st.stop()

        st.success(f"Successfully loaded: **{doc_status['filename']}** | Pages/Sections: {doc_status['pages_processed']}")

        # Run V3 Chunking Engine
        chunks = chunk_document_text(
            cleaned_text=doc_status["cleaned_text"],
            filename=doc_status["filename"],
            chunk_size=chunk_size,
            overlap=chunk_overlap
        )

        # Metrics KPI Dashboard
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Pages / Sections", doc_status["pages_processed"])
        col2.metric("Raw Characters", f"{len(doc_status['raw_text']):,}")
        col3.metric("Cleaned Characters", f"{len(doc_status['cleaned_text']):,}")
        col4.metric("Total Chunks Generated", len(chunks))

        st.divider()

        # 4. The Unified Workspace (Tabs)
        tab1, tab2, tab3, tab4 = st.tabs([
            "🧩 Document Chunks (V3 RAG Prep)", 
            "📝 AI Document Summarizer", 
            "🔍 Insight & Entity Extractor", 
            "📄 Document Text Inspector"
        ])

        # --- TAB 1: CHUNKS & METADATA (V3 PIPELINE) ---
        with tab1:
            st.subheader(f"Generated Chunks ({len(chunks)})")
            st.caption("Each chunk retains structural metadata (Source File, Chunk ID, Length) formatted for downstream Vector DB Storage.")
            
            for idx, chunk in enumerate(chunks):
                with st.expander(f"Chunk #{chunk['chunk_id']} | Length: {chunk['content_length']} chars"):
                    st.json({
                        "chunk_id": chunk["chunk_id"],
                        "source_file": chunk["source_file"],
                        "content_length": chunk["content_length"]
                    })
                    st.text_area("Chunk Content", value=chunk["content"], height=120, key=f"chunk_view_{idx}")

        # --- TAB 2: AI DOCUMENT SUMMARIZER ---
        with tab2:
            st.subheader("Generate a Structured AI Summary")
            summary_format = st.radio("Select Output Format:", ["Executive Summary", "Bullet Points", "Key Action Items"], horizontal=True)
            
            if st.button("Generate Summary", type="primary"):
                with st.spinner("Analyzing and summarizing document..."):
                    prompt = f"""
                    You are an expert technical reader and summarizer.
                    Analyze the following document and provide a summary formatted as: {summary_format}.
                    Keep the output professional, concise, and focused on the most critical information.

                    Document Text:
                    {doc_status['cleaned_text'][:15000]}
                    """
                    
                    try:
                        response = model.generate_content(prompt)
                        with st.container(border=True):
                            st.markdown(response.text)
                    except Exception as e:
                        st.error(f"Summarization failed: {e}")

        # --- TAB 3: INSIGHT & ENTITY EXTRACTOR ---
        with tab3:
            st.subheader("Extract Specific Entities or Insights")
            extraction_target = st.text_input("What would you like to extract? (e.g., 'Names and Dates', 'Financial Metrics', 'Software Tools', 'Actionable Items')")
            
            if st.button("Extract Insights", type="primary"):
                if extraction_target:
                    with st.spinner(f"Extracting {extraction_target}..."):
                        prompt = f"""
                        You are an elite data extraction AI.
                        Review the following document and extract the following specific information: {extraction_target}.
                        Format your response strictly as a clean Markdown table. If the requested information is not present, state that clearly.

                        Document Text:
                        {doc_status['cleaned_text'][:15000]}
                        """
                        
                        try:
                            response = model.generate_content(prompt)
                            with st.container(border=True):
                                st.markdown(response.text)
                        except Exception as e:
                            st.error(f"Extraction failed: {e}")
                else:
                    st.warning("Please specify what you want to extract.")

        # --- TAB 4: RAW & CLEANED TEXT PREVIEW ---
        with tab4:
            st.subheader("Extracted Text Viewer")
            view_mode = st.radio("Select View Mode:", ["Normalized & Cleaned Text", "Raw Extracted Text"], horizontal=True)
            
            if view_mode == "Normalized & Cleaned Text":
                st.text_area("Cleaned Content", doc_status["cleaned_text"], height=350, disabled=True)
            else:
                st.text_area("Raw Content", doc_status["raw_text"], height=350, disabled=True)

else:
    st.info("Please upload a PDF, DOCX, or TXT file to begin knowledge extraction.")