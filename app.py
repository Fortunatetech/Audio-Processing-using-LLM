import streamlit as st
from helper_function import annotate_audio

def main():
    st.title("Audio Annotation for Prosodic Features")
    st.write("Upload an audio file to have its prosodic features annotated using an OpenAI multimodal model with reasoning capability.")

    audio_file = st.file_uploader("Choose an audio file", type= None)
    
    if audio_file is not None:
        st.audio(audio_file, format='audio/wav')
        st.info("Annotating audio, please wait...")
        annotation_output = annotate_audio(audio_file)
        st.markdown("### Annotation Output")
        st.text(annotation_output)

if __name__ == '__main__':
    main()