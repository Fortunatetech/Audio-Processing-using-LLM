import streamlit as st
import google.generativeai as genai
import tempfile
import os
import mimetypes  

# Configure Gemini
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

PROMPT = """You are an expert in annotating prosodic features in speech.

Guidelines for Annotating Prosodic Features 

For each given speech sample, consider the following:

•	Identify the Dominant Pattern: Determine the most prevalent pattern in intonation, tempo, loudness, and pause type throughout the utterance.

•	Label Independently: Annotate each feature (intonation, tempo, loudness, pauses) separately before considering their combined effect.

•	Consider Context: The intended emotion or communicative purpose (question, statement, narrative, etc.) often informs which prosodic features are emphasized.

•	Use Consistent Criteria: Develop and follow clear definitions for what qualifies as “short” vs. “long” pauses, “slow” vs. “fast” tempo, etc.

•	Note Variations: In longer utterances, note any transitions or shifts (e.g., a rising tempo that then decelerates) as separate labels if relevant.

•	Avoid Over-Specification: While detailed labeling is useful, ensure the labels capture the essential prosodic elements without becoming overly fragmented.

Note: Your response needs to be on the basis of your observation skills and linguistic knowledge after carefully listening to the audio.

Definitions
Tempo (Speech Rate): Tempo refers to the speed at which speech is delivered. It affects the perception of urgency, thoughtfulness, or emotional intensity. Variations in tempo can also signal changes in narrative or conversational dynamics.
Loudness (Intensity): Loudness is the perceived volume or intensity of the speech. Changes in loudness help convey emphasis, emotion, and dynamics within the speech.
Intonation: Intonation is the pattern of pitch variation across phrases or sentences. It carries meaning beyond the words, indicating the speaker’s attitude, emotion, or intent (e.g., questioning, asserting, or expressing uncertainty)
Pauses: Pauses are the breaks or silences inserted within or between utterances. They help structure the speech, indicate boundaries between thoughts or phrases, and add dramatic or reflective effects.

In English, pauses play a crucial role in speech and writing. There are four main types of pauses:  

1. *Grammatical Pauses* (Punctuation Pauses)  
   - These occur due to punctuation marks such as commas, periods, colons, and semicolons.  
   - *Example*: "She loves reading, writing, and painting." (Pause at the comma)  

2. *Rhetorical Pauses*  
   - These are used for emphasis or dramatic effect, even if there is no punctuation.  
   - *Example*: "And then... everything changed."  

3. *Breath Pauses*  
   - These occur naturally when speaking to take a breath, often at logical breaks in sentences.  
   - *Example*: "After a long day at work, (pause for breath) I just want to relax."  

4. *Emotional or Hesitation Pauses*  
   - These are used when a speaker is unsure, thinking, or expressing strong emotions.  
   - *Example*: "Well... I’m not sure about that."  

Each type of pause serves a different purpose, helping to convey meaning, structure, and emotion in communication.


Note: You need to describe these parameters in a very concise manner, Meaning; The description should not be more than one line.

Study the instructions very well to understand how to answer the following section in the image above. 

Output Format:
Provide your annotation using the exact format below (each feature must be described in one concise line):

Tempo (Speech Rate): <annotation>
Loudness (Intensity): <annotation>
Intonation: <annotation>
Pauses: <annotation>

Audio 1
Tempo (Speech Rate): Moderate, with a slight increase towards the end.
Loudness (Intensity): Generally soft, with occasional emphasis on key words.
Intonation: Predominantly falling contour, with subtle rises on stressed phrases.
Pauses: Short, rhetorical and infrequent, primarily occurring before important points.

Audio 2
Tempo (Speech Rate): Moderately slow, with a brief quickening in the middle.
Loudness (Intensity): Generally moderate, with a slight increase on emphasized words.
Intonation: Mostly flat, with occasional rises signaling mild emphasis.
Pauses: Short breathe pauses yet strategically placed before key phrases.

Audio 3:
Tempo (Speech Rate):moderately slow in the beginning and slightly quick towards the end 
Loudness(intensity):Generally moderate, with slight increase on emphasized words
Intonation:mostly flat, with occasional rise signalling mild emphasis
Pauses: short, rhetorical and strategic pauses placed before key emphasis

Audio 4:
Tempo (Speech Rate):moderately quick in the beginning and slightly slow towards the end
Loudness(intensity):Generally moderate, with slight increase on emphasized words
Intonation:mostly flat, with occasional rise signalling mild emphasis
Pauses:short breathe breaks placed before key emphasis

Audio 5
Tempo (Speech Rate): Steady moderate pace with subtle rhythmic shifts.
Loudness (Intensity): Consistently moderate, with brief accentuations on key words.
Intonation: Mainly level with gentle inflections that underscore emphasis.
Pauses: Evenly breathe spaced short rhetorical pauses, occasionally lengthening to mark transitions.

Audio 6
Tempo (Speech Rate):steady moderate paced with slight increase when emphasising words.
Loudness(intensity):generally moderate , with brief accentuations on key words
Intonation:mostly flat and slight rise when expressing key words
Pauses:very short breathe paced pauses, placed before emphasis

Audio 7
Tempo (Speech Rate):moderately paced with slight increase when emphasising words
Loudness(intensity):Generally moderate even when emphasising words
Intonation:mostly urgent and expressive to emphasis certain words
Pauses: evenly paced breathe pauses and occasional rhetorical pause towards the end.

Instructions Recap:
- Analyze independently: Evaluate each prosodic feature separately before considering their combined effect.
- Context matters: Let the intended emotion or communicative purpose guide your emphasis on certain features.
- Consistency is key: Use clear definitions for terms (e.g., what qualifies as “short” vs. “long” pauses) and apply them uniformly.
- Note variations: For longer utterances, mark any significant transitions (e.g., changes in tempo) as part of your annotation.
- Keep it concise: Each description should be one line only.

"""

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
st.title("Audio Prosodic Analyzer")
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












# import streamlit as st
# import google.generativeai as genai
# import tempfile
# import os

# # Configure Gemini
# genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# PROMPT = """You are an expert in annotating prosodic features in speech.

# Guidelines for Annotating Prosodic Features 

# For each given speech sample, consider the following:

# •	Identify the Dominant Pattern: Determine the most prevalent pattern in intonation, tempo, loudness, and pause type throughout the utterance.

# •	Label Independently: Annotate each feature (intonation, tempo, loudness, pauses) separately before considering their combined effect.

# •	Consider Context: The intended emotion or communicative purpose (question, statement, narrative, etc.) often informs which prosodic features are emphasized.

# •	Use Consistent Criteria: Develop and follow clear definitions for what qualifies as “short” vs. “long” pauses, “slow” vs. “fast” tempo, etc.

# •	Note Variations: In longer utterances, note any transitions or shifts (e.g., a rising tempo that then decelerates) as separate labels if relevant.

# •	Avoid Over-Specification: While detailed labeling is useful, ensure the labels capture the essential prosodic elements without becoming overly fragmented.

# Note: Your response needs to be on the basis of your observation skills and linguistic knowledge after carefully listening to the audio.

# Definitions
# Tempo (Speech Rate): Tempo refers to the speed at which speech is delivered. It affects the perception of urgency, thoughtfulness, or emotional intensity. Variations in tempo can also signal changes in narrative or conversational dynamics.
# Loudness (Intensity): Loudness is the perceived volume or intensity of the speech. Changes in loudness help convey emphasis, emotion, and dynamics within the speech.
# Intonation: Intonation is the pattern of pitch variation across phrases or sentences. It carries meaning beyond the words, indicating the speaker’s attitude, emotion, or intent (e.g., questioning, asserting, or expressing uncertainty)
# Pauses: Pauses are the breaks or silences inserted within or between utterances. They help structure the speech, indicate boundaries between thoughts or phrases, and add dramatic or reflective effects.

# In English, pauses play a crucial role in speech and writing. There are four main types of pauses:  

# 1. *Grammatical Pauses* (Punctuation Pauses)  
#    - These occur due to punctuation marks such as commas, periods, colons, and semicolons.  
#    - *Example*: "She loves reading, writing, and painting." (Pause at the comma)  

# 2. *Rhetorical Pauses*  
#    - These are used for emphasis or dramatic effect, even if there is no punctuation.  
#    - *Example*: "And then... everything changed."  

# 3. *Breath Pauses*  
#    - These occur naturally when speaking to take a breath, often at logical breaks in sentences.  
#    - *Example*: "After a long day at work, (pause for breath) I just want to relax."  

# 4. *Emotional or Hesitation Pauses*  
#    - These are used when a speaker is unsure, thinking, or expressing strong emotions.  
#    - *Example*: "Well... I’m not sure about that."  

# Each type of pause serves a different purpose, helping to convey meaning, structure, and emotion in communication.


# Note: You need to describe these parameters in a very concise manner, Meaning; The description should not be more than one line.

# Study the instructions very well to understand how to answer the following section in the image above. 

# Output Format:
# Provide your annotation using the exact format below (each feature must be described in one concise line):

# Tempo (Speech Rate): <annotation>
# Loudness (Intensity): <annotation>
# Intonation: <annotation>
# Pauses: <annotation>

# Audio 1
# Tempo (Speech Rate): Moderate, with a slight increase towards the end.
# Loudness (Intensity): Generally soft, with occasional emphasis on key words.
# Intonation: Predominantly falling contour, with subtle rises on stressed phrases.
# Pauses: Short, rhetorical and infrequent, primarily occurring before important points.

# Audio 2
# Tempo (Speech Rate): Moderately slow, with a brief quickening in the middle.
# Loudness (Intensity): Generally moderate, with a slight increase on emphasized words.
# Intonation: Mostly flat, with occasional rises signaling mild emphasis.
# Pauses: Short breathe pauses yet strategically placed before key phrases.

# Audio 3:
# Tempo (Speech Rate):moderately slow in the beginning and slightly quick towards the end 
# Loudness(intensity):Generally moderate, with slight increase on emphasized words
# Intonation:mostly flat, with occasional rise signalling mild emphasis
# Pauses: short, rhetorical and strategic pauses placed before key emphasis

# Audio 4:
# Tempo (Speech Rate):moderately quick in the beginning and slightly slow towards the end
# Loudness(intensity):Generally moderate, with slight increase on emphasized words
# Intonation:mostly flat, with occasional rise signalling mild emphasis
# Pauses:short breathe breaks placed before key emphasis

# Audio 5
# Tempo (Speech Rate): Steady moderate pace with subtle rhythmic shifts.
# Loudness (Intensity): Consistently moderate, with brief accentuations on key words.
# Intonation: Mainly level with gentle inflections that underscore emphasis.
# Pauses: Evenly breathe spaced short rhetorical pauses, occasionally lengthening to mark transitions.

# Audio 6
# Tempo (Speech Rate):steady moderate paced with slight increase when emphasising words.
# Loudness(intensity):generally moderate , with brief accentuations on key words
# Intonation:mostly flat and slight rise when expressing key words
# Pauses:very short breathe paced pauses, placed before emphasis

# Audio 7
# Tempo (Speech Rate):moderately paced with slight increase when emphasising words
# Loudness(intensity):Generally moderate even when emphasising words
# Intonation:mostly urgent and expressive to emphasis certain words
# Pauses: evenly paced breathe pauses and occasional rhetorical pause towards the end.

# Instructions Recap:
# - Analyze independently: Evaluate each prosodic feature separately before considering their combined effect.
# - Context matters: Let the intended emotion or communicative purpose guide your emphasis on certain features.
# - Consistency is key: Use clear definitions for terms (e.g., what qualifies as “short” vs. “long” pauses) and apply them uniformly.
# - Note variations: For longer utterances, mark any significant transitions (e.g., changes in tempo) as part of your annotation.
# - Keep it concise: Each description should be one line only.

# """

# def analyze_audio(audio_bytes, file_extension):
#     """Analyze audio using Gemini's latest file handling"""
#     model = genai.GenerativeModel('models/gemini-2.0-flash-thinking-exp')
    
#     # Create temporary audio file
#     with tempfile.NamedTemporaryFile(suffix=f".{file_extension}", delete=False) as tmpfile:
#         tmpfile.write(audio_bytes)
#         tmp_path = tmpfile.name
    
#     try:
#         # Upload file to Gemini
#         audio_file = genai.upload_file(tmp_path)
        
#         response = model.generate_content(
#             contents=[audio_file, PROMPT],
#             request_options={"timeout": 300}
#         )
        
#         return response.text
#     finally:
#         # Cleanup temporary files
#         os.unlink(tmp_path)
#         if audio_file:
#             genai.delete_file(audio_file.name)

# # Streamlit UI
# st.title("Audio Prosodic Analyzer")
# uploaded_file = st.file_uploader("Upload audio", type=["mp3", "wav", "flac"])

# if uploaded_file:
#     st.audio(uploaded_file)
    
#     if st.button("Analyze"):
#         with st.spinner("Analyzing with Gemini..."):
#             try:
#                 # Get file details
#                 audio_bytes = uploaded_file.read()
#                 file_ext = uploaded_file.name.split(".")[-1].lower()
                
#                 # Get analysis
#                 analysis = analyze_audio(audio_bytes, file_ext)
                
#                 # Display results
#                 st.subheader("Analysis Results")
#                 for line in analysis.split("\n"):
#                     if ":" in line:
#                         feature, desc = line.split(":", 1)
#                         st.markdown(f"**{feature.strip()}**")
#                         st.write(desc.strip())
#                         st.divider()
                        
#                 # Show raw response
#                 with st.expander("View Raw Analysis"):
#                     st.code(analysis)
                    
#             except Exception as e:
#                 st.error(f"Analysis failed: {str(e)}")


