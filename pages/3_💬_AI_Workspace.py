import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv

# 1. Page Configuration
st.set_page_config(
    page_title="AI Workspace | PromptForge",
    page_icon="🤖",
    layout="wide"
)

load_dotenv()

# 2. Header & Controls
col1, col2 = st.columns([8, 2])
with col1:
    st.title("🤖 AI Workspace")
    st.markdown("Your intelligent assistant for ad-hoc queries, coding, and brainstorming.")
with col2:
    st.write("") # Spacing
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.chat_session = None
        st.session_state.messages = []
        st.rerun()
        
st.divider()

# 3. Configure Gemini
try:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel('gemini-2.5-flash')
except Exception as e:
    st.error("Failed to configure Gemini API. Please check your .env file.")
    st.stop()

# 4. Initialize Session State (Memory)
if "chat_session" not in st.session_state or st.session_state.chat_session is None:
    st.session_state.chat_session = model.start_chat(history=[])
if "messages" not in st.session_state:
    st.session_state.messages = []

# 5. Render Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 6. Chat Input & Processing
if prompt := st.chat_input("Ask PromptForge AI anything..."):
    
    # Display user message instantly
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # Save user message to memory
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Send message to Gemini chat session (automatically handles history)
                response = st.session_state.chat_session.send_message(prompt)
                st.markdown(response.text)
                
                # Save AI response to memory
                st.session_state.messages.append({"role": "assistant", "content": response.text})
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")