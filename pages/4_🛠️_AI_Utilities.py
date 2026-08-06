import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv

# 1. Page Configuration
st.set_page_config(
    page_title="AI Utilities | PromptForge",
    page_icon="🛠️",
    layout="wide"
)

load_dotenv()

# Configure Gemini API
try:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel('gemini-2.5-flash')
except Exception as e:
    st.error("Failed to configure Gemini API. Please check your .env file.")
    st.stop()

st.title("🛠️ AI Utilities")
st.markdown("A unified suite of lightweight productivity tools for text transformation and automated response drafting.")
st.divider()

# 2. Tabbed Architecture
tab1, tab2 = st.tabs(["🔄 Text Transformer", "📧 Response Generator"])

# --- TAB 1: TEXT TRANSFORMER ---
with tab1:
    st.subheader("Transform & Refine Text")
    
    col_opts1, col_opts2 = st.columns(2)
    with col_opts1:
        transform_type = st.selectbox(
            "Transformation Type", 
            ["Rewrite", "Translate", "Tone Change", "Grammar & Proofreading", "Format Conversion", "Summarize Briefly"]
        )
    
    with col_opts2:
        # Dynamic configuration based on transformation selected
        if transform_type == "Translate":
            target_param = st.selectbox("Target Language", ["Spanish", "French", "German", "Hindi", "Japanese", "Mandarin", "Marathi"])
        elif transform_type == "Tone Change":
            target_param = st.selectbox("Desired Tone", ["Professional", "Casual", "Persuasive", "Empathetic", "Executive", "Friendly"])
        elif transform_type == "Format Conversion":
            target_param = st.selectbox("Target Format", ["Bullet Points", "Markdown Table", "JSON", "Numbered List", "HTML"])
        else:
            target_param = None

    input_text = st.text_area("Input Text", height=180, placeholder="Paste or type text to transform...")
    
    # Text Statistics Bar
    if input_text:
        word_cnt = len(input_text.split())
        char_cnt = len(input_text)
        st.caption(f"📊 **Input Stats:** {word_cnt} words | {char_cnt} characters")

    if st.button("Transform Text", type="primary", key="btn_transform"):
        if input_text:
            with st.spinner("Processing text..."):
                param_str = f" Target: {target_param}." if target_param else ""
                prompt = f"Perform operation: {transform_type}.{param_str}\n\nText:\n{input_text}"
                try:
                    response = model.generate_content(prompt)
                    st.markdown("### Transformed Output")
                    with st.container(border=True):
                        st.write(response.text)
                    
                    # File Export Utility
                    st.download_button(
                        label="⬇️ Download Output (TXT)",
                        data=response.text,
                        file_name="transformed_text.txt",
                        mime="text/plain"
                    )
                except Exception as e:
                    st.error(f"Transformation failed: {e}")
        else:
            st.warning("Please enter text to transform.")

# --- TAB 2: RESPONSE GENERATOR ---
with tab2:
    st.subheader("Contextual Response Drafts")
    
    col_res1, col_res2 = st.columns(2)
    with col_res1:
        response_context = st.selectbox(
            "Context / Platform", 
            ["Professional Email", "Customer Support Ticket", "Social Media Comment", "Internal Slack/Teams Message", "LinkedIn Message"]
        )
    with col_res2:
        response_tone = st.selectbox(
            "Response Tone",
            ["Professional & Courteous", "Direct & Concise", "Empathetic & Warm", "Firm & Formal"]
        )

    original_message = st.text_area("Original Message / Inquiry", height=140, placeholder="Paste the incoming message here...")
    key_points = st.text_input("Key Points to Include", placeholder="e.g., Issue fixed, meeting confirmed for 3 PM, attached guide")

    if st.button("Generate Response", type="primary", key="btn_response"):
        if original_message:
            with st.spinner("Drafting response..."):
                prompt = f"""
                Draft a response for platform: {response_context}.
                Tone: {response_tone}.
                Include key points: {key_points if key_points else 'N/A'}.

                Original Message:
                {original_message}
                """
                try:
                    response = model.generate_content(prompt)
                    st.markdown("### Drafted Response")
                    with st.container(border=True):
                        st.write(response.text)
                    
                    # File Export Utility
                    st.download_button(
                        label="⬇️ Download Draft (TXT)",
                        data=response.text,
                        file_name="response_draft.txt",
                        mime="text/plain"
                    )
                except Exception as e:
                    st.error(f"Generation failed: {e}")
        else:
            st.warning("Please provide the original message to respond to.")