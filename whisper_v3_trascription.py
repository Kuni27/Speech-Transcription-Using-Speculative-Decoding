import streamlit as st
import torch
from transformers import pipeline, AutoProcessor, AutoModelForSpeechSeq2Seq
from datasets import Audio
import time

# Title and description
st.title("Whisper v3 Transcription")
st.write("Upload an audio file to transcribe using the Whisper v3 model.")

# File upload
audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a"])

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load Whisper v3 model and processor
@st.cache_resource
def load_whisper_pipeline():
    model_id = "openai/whisper-large-v3"

    # Load model
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id).to(device)
    processor = AutoProcessor.from_pretrained(model_id)

    # Create ASR pipeline
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=15,
        batch_size=1, 
        device=device,
    )
    return pipe

# Load the pipeline
whisper_pipeline = load_whisper_pipeline()

# Process and transcribe
if audio_file is not None:
    st.audio(audio_file, format="audio/wav")

    if st.button("Transcribe"):
        st.write("Processing audio...")
        try:
            # Preprocess the audio using datasets Audio or torchaudio
            audio_data = Audio(sampling_rate=16000).decode_example({"path": audio_file.name, "bytes": audio_file.read()})

            # Transcription with timing
            st.write("Running transcription...")
            start_time = time.time()
            result = whisper_pipeline(audio_data["array"])
            transcription_time = time.time() - start_time

            # Display results
            st.write("Transcription:")
            st.text(result["text"])
            st.write(f"Time taken: {transcription_time:.2f} seconds")
        except Exception as e:
            st.error(f"An error occurred: {e}")
