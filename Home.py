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
        <div class="hero-title">🤖 PromptForge AI</div>
        <div class="hero-subtitle">Data & Knowledge Intelligence Platform</div>
    </div>
""", unsafe_allow_html=True)

st.markdown("---")

# --- Dashboard Grid ---
st.subheader("Platform Modules")
st.write("Select a module to begin your workflow.")
st.write("") # Add a little breathing room

# Switched to 2 columns since we have 4 main modules now (creates a clean 2x2 grid)
col1, col2 = st.columns(2)

# Updated names for maximum professional appeal
tools = [
    ("📊 Data Intelligence", "pages/1_📊_Data_Hub.py"),
    ("📚 Knowledge Intelligence", "pages/2_📚_Knowledge_Hub.py"),
    ("🤖 AI Workspace", "pages/3_💬_AI_Workspace.py"),
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
st.write("PromptForge AI is an advanced Intelligence Platform designed to bridge the gap between raw data and actionable business insights. We empower users to automate Exploratory Data Analysis (EDA), extract knowledge from documents, and generate executive summaries using specialized AI workflows.")

st.write("") # Spacing

col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Core Mechanics")
    st.markdown("""
    * **Automated Profiling:** Instant EDA and KPI generation.
    * **Dynamic Visualizations:** Interactive Plotly charting.
    * **AI Insights:** Executive summaries from raw datasets.
    * **Context Isolation:** Protecting data integrity via Prompt Engineering.
    """)
with col_b:
    st.subheader("Technology Stack")
    st.markdown("""
    * **AI Engine:** Google Gemini 2.5 Flash
    * **Frontend:** Streamlit Python
    * **Data Processing:** Pandas & Plotly
    * **Outputs:** JSON, Markdown, & Interactive Dashboards
    """)

st.markdown("---")
st.markdown("<p style='text-align: center; color: gray; font-size: 14px;'>© 2026 PromptForge AI | Version 2.0.0</p>", unsafe_allow_html=True)