import streamlit as st
import base64
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# System prompt with annotation guidelines
SYSTEM_PROMPT = """
You are an expert in prosodic analysis. Analyze audio for these features:

1. Tempo (Speech Rate): Speed of speech delivery
2. Loudness (Intensity): Perceived volume/emphasis
3. Intonation: Pitch variation patterns
4. Pauses: Break/silence characteristics

Guidelines:
- Identify dominant patterns first
- Label features independently but consider combined effects
- Note variations and transitions
- Use consistent criteria for classifications
- Keep descriptions concise (one line per feature)

Respond EXACTLY in this format:
Tempo (Speech Rate): [Description]
Loudness (Intensity): [Description]
Intonation: [Description]
Pauses: [Description]
"""

def analyze_audio(audio_data_uri):
    """Send audio to OpenAI API for analysis"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "audio", "audio": audio_data_uri},
            {"type": "text", "text": "Analyze prosodic features following guidelines."}
        ]}
    ]
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=300
    )
    return response.choices[0].message.content

# Streamlit UI
st.title("Audio Prosodic Feature Analyzer")
st.write("Upload an audio file (MP3/WAV) for prosodic analysis")

uploaded_file = st.file_uploader("Choose audio file", type=None)

if uploaded_file:
    # Display audio player
    st.audio(uploaded_file, format=f'audio/{uploaded_file.name.split(".")[-1]}')
    
    if st.button("Analyze Prosodic Features"):
        with st.spinner("Analyzing audio..."):
            try:
                # Prepare audio data URI
                file_ext = uploaded_file.name.split(".")[-1].lower()
                mime_type = f"audio/{file_ext}" if file_ext in ["wav", "mp3"] else "audio/mpeg"
                audio_bytes = uploaded_file.getvalue()
                b64_audio = base64.b64encode(audio_bytes).decode("utf-8")
                data_uri = f"data:{mime_type};base64,{b64_audio}"

                # Get analysis from OpenAI
                analysis = analyze_audio(data_uri)
                
                # Display results
                st.subheader("Analysis Results")
                lines = analysis.split("\n")
                for line in lines:
                    if ":" in line:
                        feature, desc = line.split(":", 1)
                        st.markdown(f"**{feature.strip()}**")
                        st.write(desc.strip())
                        st.divider()
                
                # Show raw response
                with st.expander("View Detailed Analysis Report"):
                    st.write(analysis)
                    
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")