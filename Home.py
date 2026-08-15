import streamlit as st

st.set_page_config(
    page_title="PromptForge AI | Home",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Hide Sidebar and center the hero section
st.markdown("""
    <style>
        /* Hides the sidebar completely */
        [data-testid="stSidebar"] {display: none;}
        
        /* Custom styling to center the main headers */
        .hero-box {
            text-align: center;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .hero-title {
            font-size: 3.5rem;
            font-weight: 800;
            margin-bottom: 0px;
        }
        .hero-subtitle {
            font-size: 1.2rem;
            opacity: 0.7;
            font-weight: 500;
            margin-top: 5px;
        }
    </style>
""", unsafe_allow_html=True)

# --- Hero Section (Centered via HTML/CSS) ---
st.markdown("""
    <div class="hero-box">
        <div class="hero-title">⚡ PromptForge AI</div>
        <div class="hero-subtitle">Enterprise Agentic Reasoning & Hybrid RAG Engine</div>
    </div>
""", unsafe_allow_html=True)

st.markdown("---")

# --- Dashboard Grid ---
st.subheader("Platform Modules")
st.write("Select a module to begin your workflow.")
st.write("") # Add a little breathing room

# 2x2 Grid for the 4 main modules
col1, col2 = st.columns(2)

# Exact paths matched to your local directory structure
tools = [
    ("📊 Data Hub (Structured SQL)", "pages/1_📊_Data_Hub.py"),
    ("📚 Knowledge Hub (Vector RAG)", "pages/2_📚_Knowledge_Hub.py"),
    ("🤖 AI Workspace (Agentic Chat)", "pages/3_💬_AI_Workspace.py"),
    ("🛠️ AI Utilities", "pages/4_🛠️_AI_Utilities.py")
]

# Create the grid
for i, (name, path) in enumerate(tools):
    col = [col1, col2][i % 2]
    with col:
        # use_container_width makes the buttons span the full column for a "card" feel
        if st.button(name, use_container_width=True):
            st.switch_page(path)

st.markdown("---")

# --- Dashboard Info ---
st.header("About PromptForge AI")
st.write("PromptForge AI is an advanced AI architecture demonstrating autonomous reasoning across Structured Database Metrics (SQL) and Unstructured Document Context (RAG). It utilizes intelligent routing, orchestrators, and reviewer nodes to prevent hallucinations and deliver verified, cited insights.")

st.write("") # Spacing

col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Core Capabilities")
    st.markdown("""
    * **Automated SQL Engineering:** Translates natural language to SQL against live PostgreSQL.
    * **Semantic RAG Engine:** Math-based chunking and vector retrieval via local ChromaDB.
    * **Agentic Routing:** AI autonomously classifies intents (SQL vs. RAG vs. Hybrid).
    * **Reviewer Node (QA):** Validates facts and enforces strict citation rules.
    """)
with col_b:
    st.subheader("Technology Stack")
    st.markdown("""
    * **AI Engine:** Google Gemini 2.5 Flash
    * **Database (SQL):** Neon Serverless PostgreSQL & SQLAlchemy
    * **Vector Store (NoSQL):** ChromaDB Persistent Client
    * **Frontend:** Streamlit Python
    """)

st.markdown("---")
st.markdown("<p style='text-align: center; color: gray; font-size: 14px;'>© 2026 PromptForge AI | Version 4.0.0</p>", unsafe_allow_html=True)