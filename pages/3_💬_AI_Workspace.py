import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from src.ai.orchestrator import process_user_query
from src.ai.reviewer import review_output
from src.utils.pdf_exporter import generate_pdf_report

# 1. Page Configuration
st.set_page_config(
    page_title="AI Workspace | PromptForge",
    page_icon="🤖",
    layout="wide"
)

load_dotenv()

# 2. Header & Conversation Controls
col1, col2 = st.columns([8, 2])
with col1:
    st.title("🤖 Agentic AI Workspace")
    st.markdown("Your intelligent assistant for conversational queries, database analysis, predictive modeling, and document research.")

with col2:
    st.write("") # Spacing
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.divider()

# 3. Sidebar Workspace Context (V5.0 Workspace Awareness)
with st.sidebar:
    st.header("⚙️ Workspace Context")
    
    # Dynamically pull active table set by Data Hub (falls back to "coffe_sales")
    default_table = st.session_state.get('active_table', 'coffe_sales')
    active_table = st.text_input(
        "Active PostgreSQL Table:", 
        value=default_table, 
        help="Target database table created in Data Hub."
    )
    
    # Dynamically pull auto-detected schema set by Data Hub (falls back to default demo schema)
    default_schema = st.session_state.get(
        'auto_schema', 
        "hour_of_day (int), cash_type (str), money (float), coffee_name (str)"
    )
    
    table_schema = st.text_area(
        "Table Schema Summary:", 
        value=default_schema,
        height=140,
        help="Column names, data types, and semantic roles automatically mapped from Data Hub."
    )

# 4. Initialize Session Memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# 5. Render Previous Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # Display Router badge if present
        if "route_info" in message:
            st.caption(message["route_info"])
        
        st.markdown(message["content"])
        
        # Render expanders for SQL, RAG, or ML evidence if saved in history
        if message.get("sql_code"):
            with st.expander("📊 Executed SQL Query"):
                st.code(message["sql_code"], language="sql")
                if "sql_data" in message:
                    st.dataframe(message["sql_data"])
                    
        if message.get("rag_sources"):
            with st.expander("📚 Retrieved Document Evidence"):
                for idx, src in enumerate(message["rag_sources"]):
                    st.caption(f"**Source {idx+1}: {src['meta']['source_file']} (Chunk {src['meta']['chunk_id']})**")
                    st.write(src["doc"])
                    st.divider()
                    
        if message.get("ml_result"):
            with st.expander("🔮 View ML Engine Results"):
                st.json(message["ml_result"])

# 6. Chat Input & Agentic Processing
if prompt := st.chat_input("Ask PromptForge AI anything across your data and documents..."):
    
    # Render user prompt
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Render assistant response using V5 Orchestrator and Reviewer
    with st.chat_message("assistant"):
        with st.spinner("Routing query and executing pipeline..."):
            result = process_user_query(
                query=prompt,
                table_name=active_table if active_table else None,
                schema_info=table_schema
            )
            
            if not result["success"]:
                error_msg = f"Execution Error: {result['error']}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
            else:
                # Format Router decision banner
                route_labels = {
                    "sql": "📊 SQL Database Query", 
                    "rag": "📚 Knowledge Base RAG", 
                    "ml": "🔮 Predictive ML Pipeline",
                    "hybrid": "🔀 Hybrid Pipeline (SQL + RAG)"
                }
                route_badge = f"**Route Chosen:** `{route_labels.get(result['route'], result['route'])}` | *{result['reasoning']}*"
                st.caption(route_badge)

                # --- V4.9 THE REVIEWER NODE ---
                with st.spinner("Reviewer Node validating data accuracy..."):
                    review = review_output(prompt, result)
                    
                    if review["success"]:
                        if review["is_valid"]:
                            st.success(f"✅ **Reviewer Approved:** {review['feedback']}")
                        else:
                            st.warning(f"⚠️ **Reviewer Intervened:** {review['feedback']}")
                        
                        final_text = review["final_approved_text"]
                    else:
                        st.error("Reviewer Node failed. Displaying raw output.")
                        final_text = result["final_synthesis"]
                
                # Display the final, reviewer-approved text
                st.markdown(final_text)
                
                # --- V5.0 PDF EXPORT FEATURE ---
                pdf_bytes = generate_pdf_report(
                    query=prompt, 
                    synthesis=final_text, 
                    route=result["route"]
                )
                
                st.download_button(
                    label="⬇️ Download Executive Brief (PDF)",
                    data=pdf_bytes,
                    file_name="PromptForge_Executive_Brief.pdf",
                    mime="application/pdf",
                    type="primary"
                )
                
                # Prepare history dictionary
                msg_data = {
                    "role": "assistant",
                    "content": final_text,
                    "route_info": route_badge
                }
                
                # Render SQL execution expander
                if result.get("sql_result") and "data" in result["sql_result"]:
                    st.code(result["sql_result"]["sql"], language="sql")
                    with st.expander("📊 View SQL Execution Results"):
                        st.dataframe(result["sql_result"]["data"])
                    msg_data["sql_code"] = result["sql_result"]["sql"]
                    msg_data["sql_data"] = result["sql_result"]["data"]
                    
                # Render RAG source evidence expander
                if result.get("rag_result"):
                    with st.expander("📚 View Document Sources & Retrieval Evidence"):
                        for idx, src in enumerate(result["rag_result"]["sources"]):
                            st.caption(f"**Source {idx+1}: {src['meta']['source_file']} (Chunk {src['meta']['chunk_id']})**")
                            st.write(src["doc"])
                            st.divider()
                    msg_data["rag_sources"] = result["rag_result"]["sources"]
                    
                # Render ML execution expander
                if result.get("ml_result"):
                    with st.expander("🔮 View ML Engine Results"):
                        st.json(result["ml_result"])
                    msg_data["ml_result"] = result["ml_result"]

                # Append enriched response to chat state
                st.session_state.messages.append(msg_data)