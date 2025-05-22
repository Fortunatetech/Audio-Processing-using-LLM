import streamlit as st
import google.generativeai as genai
import tempfile
import os
import mimetypes  
import time 

# Configure Gemini
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

@st.cache_data
def load_prompt() -> str:
    return """You are an expert in annotating Paralinguistic Information in an audio clip. Listen and study the audio clip carefully. 
              You task is to List out the Fillers words, Non-Verbal Expressions, and Non-Verbal Descriptors found in the audio clip. 
              Below are the list of section:

              For Fillers words:
               Uh, Huh, Hmm, Ahem, Ahaan, Aha, Ta-da, Duh, Phew, Yippee, Yay, Yuck, Er, Um, Ooh, Aah, Shh, Yikes, Whoa, Ugh, Oops, Ew, Uh-oh
               Psst, Wow, Awww, Uh-huh, Mm-Hmm.

              For Non-Verbal Expressions:
               Sighs, Coughs, Laughs, Chuckles, Scoffs, Panting, Pause, Gasping, Groans, Sobbing, Giggling, Growling, Whimpering, Clears throat
               Grumbling, Sniffing, Hissing, Snarling, Grunting, Noise

              For Non-Verbal Descriptors:
              Emphasis, Stutter, Laughter, Exclaiming, Giggling, Sobbing, Slurring, Fast, Slow, Noise

              Note for proper annotation:
              - Your listening should start from the momment the person starts speaking from the beginning and should ends immediately the person stops talking at the end of te clip.
              - The puses could occur multiple places throuhg the entire time the person was talking. So try to figure all the positions where the pauses occured.
              - This is to avoid pauses before and after the person talking. Specify the phrase where the pause occur.

             Output Format:
              1. Transcription: Generate the trancription.
              2. Fillers words: List Fillers words found and the position the filler word(s) where they occur if not found return None.
              3. Non-Verbal Expressions: list the Non-Verbal Expressions found and position the expression where they occur in the transcription.
              4. Non-Verbal Descriptors: list the Non-Verbal Descriptors found and position the Decriptors where they occur in the Transcription.
                """
@st.cache_resource
def get_model():
    # create the model object once and reuse it
    return genai.GenerativeModel("models/gemini-2.0-flash-thinking-exp")

PROMPT = load_prompt()
MODEL = get_model()

def safe_remove(path: str, retries: int = 5, delay: float = 0.2):
    """Retry unlinking a file to avoid transient Windows locks."""
    for _ in range(retries):
        try:
            os.unlink(path)
            return
        except PermissionError:
            time.sleep(delay)
    os.unlink(path)

def analyze_audio(audio_bytes: bytes, file_extension: str) -> str:
    """Analyze audio with Gemini, specifying mime_type for file-like uploads."""
    # 1) Write bytes to temp file (closed on exit)
    with tempfile.NamedTemporaryFile(suffix=f".{file_extension}", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    audio_file = None
    try:
        # 2) Guess MIME type from extension
        mime_type, _ = mimetypes.guess_type(tmp_path)
        if mime_type is None:
            # fallback for common audio
            mime_type = f"audio/{file_extension}"
        
        # 3) Upload by path with explicit mime_type
        audio_file = genai.upload_file(tmp_path, mime_type=mime_type)

        # 4) Generate annotation
        model = genai.GenerativeModel("models/gemini-2.0-flash-thinking-exp")
        response = model.generate_content(
            contents=[audio_file, PROMPT],
            request_options={"timeout": 300}
        )
        return response.text

    finally:
        # 5) Remove Gemini’s copy, then our temp file
        if audio_file:
            genai.delete_file(audio_file.name)
            audio_file = None
        safe_remove(tmp_path)

# — Streamlit UI —
st.title("Audio Paralinguistic Analyzer")
uploaded_file = st.file_uploader("Upload audio", type=["mp3", "wav", "flac"])

if uploaded_file:
    st.audio(uploaded_file)

    if st.button("Analyze"):
        with st.spinner("Analyzing with Gemini..."):
            try:
                audio_bytes = uploaded_file.read()
                file_ext = uploaded_file.name.rsplit(".", 1)[-1].lower()
                analysis = analyze_audio(audio_bytes, file_ext)

                st.subheader("Analysis Results")
                for line in analysis.splitlines():
                    if ":" in line:
                        feature, desc = line.split(":", 1)
                        st.markdown(f"**{feature.strip()}**")
                        st.write(desc.strip())
                        st.divider()

                with st.expander("View Raw Analysis"):
                    st.code(analysis)

            except Exception as e:
                st.error(f"Analysis failed: {e}")


