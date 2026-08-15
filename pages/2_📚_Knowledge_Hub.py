import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Import Core Backend Document Pipelines from src/
from src.documents.loaders import extract_text_from_file
from src.documents.chunking import chunk_document_text
from src.documents.vectorstore import store_chunks_in_chroma, semantic_search
from src.ai.rag import generate_rag_answer

# 1. Page Configuration
st.set_page_config(page_title="Knowledge Hub | PromptForge", page_icon="📚", layout="wide")
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
your knowledge base for AI-driven summarization, entity extraction, and Version 5.0 RAG systems.
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
        # Run Ingestion Engine
        doc_status = extract_text_from_file(uploaded_doc, uploaded_doc.name)
        
        if not doc_status["success"]:
            st.error(f"Extraction Error: {doc_status['error']}")
            st.stop()

        st.success(f"Successfully loaded: **{doc_status['filename']}** | Pages/Sections: {doc_status['pages_processed']}")

        # Run Chunking Engine
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
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🧩 Vector DB Storage", 
            "🤖 Full RAG Chat (V5.0)", 
            "🧠 Semantic Search",
            "📝 AI Summarizer", 
            "🔍 Entity Extractor", 
            "📄 Raw Text"
        ])

        # --- TAB 1: CHUNKS & VECTOR DB ---
        with tab1:
            st.subheader(f"Generated Chunks ({len(chunks)})")
            st.caption("Review chunks before pushing them to the persistent ChromaDB Vector Store.")
            
            if st.button("💾 Push to Vector Database", type="primary"):
                with st.spinner("Embedding chunks via local engine (Rate-Limit Free) and storing in ChromaDB..."):
                    db_status = store_chunks_in_chroma(chunks)
                    if db_status["success"]:
                        st.success(f"Successfully embedded and stored {db_status['chunks_inserted']} chunks in ChromaDB!")
                    else:
                        st.error(db_status["error"])
            
            st.divider()
            
            for idx, chunk in enumerate(chunks[:10]):
                with st.expander(f"Chunk #{chunk['chunk_id']} | Length: {chunk['content_length']} chars"):
                    st.json({"chunk_id": chunk["chunk_id"], "source_file": chunk["source_file"]})
                    st.text_area("Chunk Content", value=chunk["content"], height=120, key=f"chunk_view_{idx}")
            if len(chunks) > 10:
                st.info(f"...and {len(chunks) - 10} more chunks ready for storage.")

        # --- TAB 2: FULL RAG ENGINE ---
        with tab2:
            st.subheader("🤖 Chat with your Documents")
            st.write("Ask a question. PromptForge AI will retrieve the most relevant chunks and synthesize a fully cited answer.")
            
            rag_query = st.text_input("Ask the AI about your database documents:", key="rag_input")
            
            if st.button("Generate Cited Answer", type="primary"):
                if rag_query:
                    with st.spinner("Retrieving context and synthesizing answer..."):
                        rag_status = generate_rag_answer(rag_query, top_k=25)
                        
                        if rag_status["success"]:
                            # Display the AI's answer
                            st.markdown("### 💡 AI Response")
                            st.info(rag_status["answer"])
                            
                            # Display the exact sources used to prevent hallucination
                            with st.expander("🔍 View Retrieved Sources (Evidence)"):
                                for i, source in enumerate(rag_status["sources"]):
                                    st.markdown(f"**Source {i+1}: {source['meta']['source_file']} (Chunk {source['meta']['chunk_id']})**")
                                    st.caption(source["doc"])
                                    st.divider()
                        else:
                            st.error(rag_status["error"])
                else:
                    st.warning("Please enter a question.")

        # --- TAB 3: SEMANTIC SEARCH ---
        with tab3:
            st.subheader("🔍 Query the Vector Database")
            st.write("Search the database using meaning and context, rather than exact keywords.")
            
            search_query = st.text_input("Enter a question or topic to search for:")
            top_k = st.slider("Number of results to retrieve (Top-K)", min_value=1, max_value=10, value=3)
            
            if st.button("Search Vector Space", type="primary"):
                if search_query:
                    with st.spinner("Calculating vector distances..."):
                        search_status = semantic_search(search_query, n_results=top_k)
                        
                        if search_status["success"]:
                            chroma_data = search_status["results"]
                            documents = chroma_data["documents"][0]
                            metadatas = chroma_data["metadatas"][0]
                            distances = chroma_data["distances"][0]
                            
                            st.success(f"Retrieved Top {len(documents)} most relevant chunks!")
                            
                            # Display the retrieved chunks
                            for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
                                with st.container(border=True):
                                    st.markdown(f"**Result {i + 1}** | Vector Distance: `{dist:.4f}`")
                                    st.caption(f"📄 Source: {meta['source_file']} | 🧩 Chunk ID: {meta['chunk_id']}")
                                    st.write(doc)
                        else:
                            st.error(search_status["error"])
                else:
                    st.warning("Please enter a search query.")

        # --- TAB 4: AI DOCUMENT SUMMARIZER ---
        with tab4:
            st.subheader("Generate a Structured AI Summary")
            summary_format = st.radio("Select Output Format:", ["Executive Summary", "Bullet Points", "Key Action Items"], horizontal=True)
            if st.button("Generate Summary", type="primary"):
                with st.spinner("Analyzing and summarizing document..."):
                    prompt = f"Analyze the following document and provide a summary formatted as: {summary_format}.\n\nDocument Text:\n{doc_status['cleaned_text'][:15000]}"
                    try:
                        response = model.generate_content(prompt)
                        with st.container(border=True):
                            st.markdown(response.text)
                    except Exception as e:
                        st.error(f"Summarization failed: {e}")

        # --- TAB 5: INSIGHT & ENTITY EXTRACTOR ---
        with tab5:
            st.subheader("Extract Specific Entities or Insights")
            extraction_target = st.text_input("What would you like to extract? (e.g., 'Names and Dates', 'Financial Metrics')")
            if st.button("Extract Insights", type="primary"):
                if extraction_target:
                    with st.spinner(f"Extracting {extraction_target}..."):
                        prompt = f"Extract the following specific information: {extraction_target}. Format strictly as a Markdown table.\n\nDocument Text:\n{doc_status['cleaned_text'][:15000]}"
                        try:
                            response = model.generate_content(prompt)
                            with st.container(border=True):
                                st.markdown(response.text)
                        except Exception as e:
                            st.error(f"Extraction failed: {e}")
                else:
                    st.warning("Please specify what you want to extract.")

        # --- TAB 6: RAW & CLEANED TEXT PREVIEW ---
        with tab6:
            st.subheader("Extracted Text Viewer")
            view_mode = st.radio("Select View Mode:", ["Normalized & Cleaned Text", "Raw Extracted Text"], horizontal=True)
            if view_mode == "Normalized & Cleaned Text":
                st.text_area("Cleaned Content", doc_status["cleaned_text"], height=350, disabled=True)
            else:
                st.text_area("Raw Content", doc_status["raw_text"], height=350, disabled=True)

else:
    st.info("Please upload a PDF, DOCX, or TXT file to begin knowledge extraction.")