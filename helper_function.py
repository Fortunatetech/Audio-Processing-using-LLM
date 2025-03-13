import streamlit as st
import openai
import tempfile
import os
from dotenv import load_dotenv

load_dotenv()

# Make sure to set your OpenAI API key; you can also store it in Streamlit secrets.
openai.api_key = os.getenv("OPENAI_API_KEY")

def annotate_audio(audio_file):
    """
    Process the uploaded audio file using OpenAI's multimodal model to annotate prosodic features.
    
    The function:
    • Saves the uploaded audio to a temporary file.
    • Defines a clear prompt instructing the model to analyze the audio, including a brief reasoning for each feature.
    • Calls the (pseudo) multimodal API endpoint using the updated model.
    • Returns the annotation output in the specified format.
    
    Note: Adjust the API endpoint and parameters according to the latest OpenAI documentation.
    """
    # Save the audio file temporarily for processing
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_file.read())
        tmp_path = tmp.name

    # Define the detailed prompt with reasoning instructions
    prompt = (
        "You are an expert in annotating prosodic features in speech. Listen carefully to the provided audio "
        "and analyze the following aspects:\n"
        "1. Tempo (Speech Rate): Identify the dominant pattern in speech rate, including any changes or emphasis.\n"
        "2. Loudness (Intensity): Note the overall volume and any dynamic emphasis in key words.\n"
        "3. Intonation: Determine the pitch variation pattern, noting rises and falls.\n"
        "4. Pauses: Identify the duration and frequency of pauses.\n\n"
        "Provide your output in exactly the following format:\n"
        "Tempo (Speech Rate): <annotation>\n"
        "Loudness (Intensity): <annotation>\n"
        "Intonation: <annotation>\n"
        "Pauses: <annotation>\n\n"
        "Additionally, include a brief reasoning statement for each feature (one sentence per feature) as part of your analysis."
    )

    try:
        # This API call assumes a multimodal endpoint that accepts an audio file.
        # Replace "gpt-4-multimodal" with the actual model name per the latest docs.
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert in audio analysis and prosody annotation."},
                {"role": "user", "content": prompt},
                {"role": "assistance", "audio": {"id": "file=open(tmp_path, "rb")"}}
            ],
            # The API is assumed to support file input for multimodal processing.
            
        )
        # Extract the annotation output from the API response
        output = response.choices[0].message['content']
    except Exception as e:
        st.error(f"Error processing audio: {e}")
  
    return output
