# ⚡ PromptForge AI
**Enterprise-Grade Multi-Agent Data & Knowledge Intelligence Platform**

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Google Gemini](https://img.shields.io/badge/Gemini_2.5_Flash-8E75B2?style=for-the-badge&logo=googlebard&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-316192?style=for-the-badge&logo=postgresql&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-orange?style=for-the-badge)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

PromptForge AI is an agentic data intelligence platform designed to reason across **structured databases (PostgreSQL)** and **unstructured documents (ChromaDB)** in a unified workflow. It autonomously routes queries, writes and executes safe SQL, retrieves document citations, performs predictive machine learning analytics, and validates outputs using a self-correcting Critic Agent.

---

## 📸 Platform Overview

### 1. Interactive Data & Analytics Hub
*(Upload, clean, validate, and dynamically profile structured datasets)*
<div align="center">
  <img src="assets/screenshots/01-data-analytics-dashboard.png?raw=true" alt="Data Hub Screenshot" width="800"/>
</div>

### 2. Knowledge Hub & AI Workspace
*(RAG knowledge ingestion, multi-agent hybrid reasoning, ML forecasting, and professional response generation)*
<div align="center">
  <img src="assets/screenshots/03-rag-document-chat.png?raw=true" alt="RAG Chat Screenshot" width="400"/>
  <img src="assets/screenshots/02-ai-response-generator.png?raw=true" alt="Response Generator Screenshot" width="400"/>
</div>

---

## 🧠 Multi-Agent Architecture

```mermaid
flowchart TD
    User((User Query)) --> Router{🧭 Router Agent}

    Router -->|Database| SQL[📊 Text-to-SQL Agent<br/>PostgreSQL]
    Router -->|Documents| RAG[📚 Document RAG<br/>ChromaDB]
    Router -->|Forecasting| ML[🔮 ML Engine<br/>Scikit-Learn]
    Router -->|Cross-Analysis| Hybrid[🔀 Hybrid Pipeline<br/>SQL + RAG]

    SQL --> Synth[🧠 AI Synthesizer]
    RAG --> Synth
    ML --> Synth
    Hybrid --> Synth
🧭 Router Agent (router.py): Analyzes natural language intent and dynamically classifies queries into SQL, RAG, ML, or Hybrid pipelines.

📊 Text-to-SQL Agent (sql_generator.py): Schema-aware SQL generation with parameterized execution against PostgreSQL. Includes strict read-only security validation.

📚 Document RAG Agent (rag.py & vectorstore.py): Semantic search and chunk retrieval with inline citation metadata across uploaded PDFs, DOCX, and TXT files.

🔮 Predictive ML Engine (ml_engine.py): Autonomous, code-free statistical modeling:

Time-Series Forecasting: Dynamic Ridge Regression forecasting (RidgeCV) with intelligent temporal step detection.

Anomaly Detection: Unsupervised outlier identification using IsolationForest.

🛡️ Critic Node / Reviewer Agent (reviewer.py): Strict QA layer that checks proposed answers against ground-truth contexts (SQL tables, RAG chunks, or ML arrays) using structured JSON evaluation to eliminate hallucinations.

🚀 Key Features
Fault-Tolerant Auto-Retry Logic: Built-in exponential backoff gracefully handles API rate limits (429 Quota Exceeded) in the background without crashing the user interface.

Hybrid Cross-Reasoning: Answers complex queries that require both quantitative database records and qualitative document claims (e.g., comparing actual SQL revenue against PDF report targets).

Self-Healing Dynamic Schema Mapping: Automatically identifies numeric targets (money, revenue, sales) and handles both calendar dates and integer-based time indices.

Explainability by Design: Inspect raw SQL queries, explore retrieved document chunk evidence, and view raw ML JSON payloads directly in the UI.

One-Click Executive Briefs: Compiles final, reviewer-approved insights and pipeline metadata into downloadable, formatted PDF reports.

🛠️ Technology Stack
Frontend & UI: Streamlit, Plotly, ReportLab (PDF Generation)

LLM & Reasoning: Google Gemini (Gemini 2.5 Flash API) via google-generativeai

Databases: PostgreSQL (Neon Cloud), ChromaDB (Vector Store)

Machine Learning & Analytics: Scikit-Learn, Pandas, NumPy

Architecture: Multi-Agent Orchestration, Retrieval-Augmented Generation (RAG), Zero-Shot Routing

📈 The Development Journey (v1.0 → v5.0)
v1.0 — Text-to-SQL Foundation: Natural language to PostgreSQL query generation and safe execution.

v2.0 — Data Intelligence & Analytics: Automated EDA, dynamic data profiling, and interactive Plotly visualizations.

v3.0 — Data & Document Pipelines: Ingestion, validation, cleaning, chunking, and metadata indexing pipelines.

v4.0 — Hybrid RAG & Agentic Workflow: Orchestrator architecture, vector retrieval, cross-source synthesis, and the anti-hallucination Critic Node.

v5.0 — Predictive ML & Production Polish: Integrated time-series forecasting, unsupervised anomaly detection, exponential backoff resilience, and automated PDF executive reporting.

💻 Local Installation & Setup
1. Clone the repository:

Bash
git clone [https://github.com/mehulchavan11/PromptForge-AI.git](https://github.com/mehulchavan11/PromptForge-AI.git)
cd PromptForge-AI
2. Set up a virtual environment:

Bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
3. Install dependencies:

Bash
pip install -r requirements.txt
4. Configure Environment Variables:
Create a .env file in the root directory:

Code snippet
GEMINI_API_KEY="your_google_gemini_api_key"
DATABASE_URL="postgresql://username:password@host:port/database"
5. Launch the Application:

Bash
streamlit run Home.py
👨‍💻 Author
Mehul Chavan

GitHub: @mehulchavan11


    Synth --> Critic[🛡️ Critic Node<br/>Anti-Hallucination Guardrail]
    
    Critic -->|Validates Data| PDF[📄 Executive Brief / PDF]
