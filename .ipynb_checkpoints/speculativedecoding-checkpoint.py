import streamlit as st
import time
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, AutoModelForCausalLM

# Title and description
st.title("Speculative Decoding Transcription")
st.write("Upload an audio file to transcribe using Whisper and Distilled Whisper models.")

# File upload
audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a"])

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Load models
model_id = "openai/whisper-large-v3"
st.write("Loading Whisper model...")
whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
    use_safetensors=True,
    attn_implementation="sdpa",
)
whisper_model.to(device)
processor = AutoProcessor.from_pretrained(model_id)

assistant_model_id = "mpanda27/distil-whisper-large-v3"
st.write("Loading Distilled Whisper model...")
assistant_model = AutoModelForCausalLM.from_pretrained(
    assistant_model_id,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
    use_safetensors=True,
    attn_implementation="sdpa",
)
assistant_model.to(device)

if audio_file is not None:
    st.audio(audio_file, format='audio/wav')

    # Transcription process
    st.write("Starting transcription...")

    with st.spinner("Transcribing with Whisper model..."):
        start_time_whisper = time.time()
        audio_input = processor(audio_file.read(), return_tensors="pt", sampling_rate=16000).to(device)
        whisper_output = whisper_model.generate(**audio_input)
        whisper_result = processor.batch_decode(whisper_output, skip_special_tokens=True)[0]
        whisper_time = time.time() - start_time_whisper

    st.write("Whisper Transcription:")
    st.text(whisper_result)
    st.write(f"Time taken: {whisper_time:.2f} seconds")

    with st.spinner("Transcribing with Distilled Whisper model..."):
        start_time_distilled = time.time()
        distilled_output = assistant_model.generate(**audio_input)
        distilled_result = processor.batch_decode(distilled_output, skip_special_tokens=True)[0]
        distilled_time = time.time() - start_time_distilled

    st.write("Distilled Whisper Transcription:")
    st.text(distilled_result)
    st.write(f"Time taken: {distilled_time:.2f} seconds")
