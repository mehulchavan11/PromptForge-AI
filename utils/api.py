import os
import google.generativeai as genai
from dotenv import load_dotenv

# --- Configuration & Initialization ---
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if api_key:
    genai.configure(api_key=api_key)
else:
    raise ValueError("⚠️ GEMINI_API_KEY not found in .env file. Please configure your environment variables.")

# --- Model Settings ---
# We use a low temperature (0.2) to ensure outputs are highly deterministic, 
# strictly following prompt instructions rather than being overly creative.
enterprise_config = {
    "temperature": 0.2,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 2048,
}

# Initialize the Gemini model with our custom configuration
model = genai.GenerativeModel(
    model_name='gemini-2.5-flash',
    generation_config=enterprise_config
)

# --- Core Function ---
def get_ai_response(prompt: str) -> str:
    """
    Sends a structured prompt to the Gemini API and returns the generated text.
    Designed for precision Prompt Engineering workflows.
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        # We raise the error here so that the Streamlit modules (which have try/except blocks) 
        # can catch it and display a proper red st.error() banner to the user.
        raise Exception(f"API Communication Error: {str(e)}")