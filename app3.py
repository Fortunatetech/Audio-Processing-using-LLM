import streamlit as st
import openai
import tempfile

# Set your API key
openai.api_key = st.secrets["OPENAI_API_KEY"]

SYSTEM_PROMPT = """
[Keep the same system prompt as before]
"""

def analyze_audio(file_path, filename):
    """
    Upload the audio file to OpenAI using the dedicated file endpoint and then reference it in your prompt.
    This avoids embedding the file content as base64.
    """
    # Upload file using the dedicated file upload API (adjust purpose as required by docs)
    with open(file_path, "rb") as f:
        file_upload_response = openai.files.create(file=f, purpose="multimodal")
    file_id = file_upload_response.get("id")
    
    # Use the file ID in the prompt
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"Please analyze the prosodic features of the audio file referenced by file ID {file_id}. "
            "Follow the guidelines for prosodic annotation."
        )}
    ]
    
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=300
    )
    return response.choices[0].message["content"]

# Streamlit UI
st.title("Audio Prosodic Feature Analyzer")
uploaded_file = st.file_uploader("Choose audio file", type=None)

if uploaded_file:
    st.audio(uploaded_file)
    
    if st.button("Analyze"):
        with st.spinner("Analyzing..."):
            try:
                # Save the uploaded file to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                # Get the analysis from the API using the file upload method
                analysis = analyze_audio(tmp_file_path, uploaded_file.name)
                
                st.markdown("### Annotation Output")
                st.text(analysis)
            except Exception as e:
                st.error(f"Error: {str(e)}")
