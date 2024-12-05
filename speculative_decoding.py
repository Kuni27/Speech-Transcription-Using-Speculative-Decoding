import streamlit as st
import torch
from transformers import pipeline, AutoProcessor, AutoModelForSpeechSeq2Seq, AutoModelForCausalLM
from datasets import Audio
import time

# Title and description
st.title("Speculative Decoding Transcription")
st.write("Upload an audio file to transcribe using Whisper and Distilled Whisper models.")

# File upload
audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a"])

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Load models and processor
@st.cache_resource
def load_speculative_decoding_pipeline():
    model_id = "openai/whisper-large-v3"
    assistant_model_id = "mpanda27/distil-whisper-large-v3"

    # Load main model and assistant model
    main_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, 
        torch_dtype=torch_dtype, 
        low_cpu_mem_usage=True, 
        use_safetensors=True
    ).to(device)

    assistant_model = AutoModelForCausalLM.from_pretrained(
        assistant_model_id, 
        torch_dtype=torch_dtype, 
        low_cpu_mem_usage=True, 
        use_safetensors=True
    ).to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    return main_model, assistant_model, processor

def transcribe_with_speculative_decoding(main_model, assistant_model, processor, audio_array, sampling_rate):
    # Preprocess audio
    inputs = processor(audio_array, sampling_rate=sampling_rate, return_tensors="pt").to(device)

    # Measure generation time
    start_time = time.time()
    outputs = main_model.generate(**inputs, assistant_model=assistant_model)
    generation_time = time.time() - start_time

    # Decode transcription
    transcription = processor.batch_decode(outputs, skip_special_tokens=True, normalize=True)[0]

    return transcription, generation_time

# Load the models and processor
main_model, assistant_model, processor = load_speculative_decoding_pipeline()

# Process and transcribe
if audio_file is not None:
    st.audio(audio_file, format="audio/wav")

    if st.button("Transcribe"):
        st.write("Processing audio...")
        try:
            # Preprocess the audio using datasets Audio
            audio_data = Audio(sampling_rate=16000).decode_example({"path": audio_file.name, "bytes": audio_file.read()})

            # Transcription with timing
            transcription, transcription_time = transcribe_with_speculative_decoding(
                main_model, assistant_model, processor, audio_data["array"], 16000
            )

            # Display results
            st.write("Transcription:")
            st.text(transcription)
            st.write(f"Time taken: {transcription_time:.2f} seconds")
        except Exception as e:
            st.error(f"An error occurred: {e}")
