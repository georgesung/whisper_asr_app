import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import streamlit as st
from st_audiorec import st_audiorec


# ASR model
@st.cache_resource
def load_model_pipe():
    # Hardware settings
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # Set up model & pipeline
    model_id = "distil-whisper/distil-large-v3"  # "openai/whisper-large-v3"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=256,
        chunk_length_s=30,
        batch_size=8,
        return_timestamps=False,
        torch_dtype=torch_dtype,
        device=device,
    )
    return pipe

pipe = load_model_pipe()

# Wrapper function for ASR
def transcribe_speech(audio_obj):
    output = pipe(
        audio_obj,
        generate_kwargs={
            "task": "transcribe",
            "language": "english",  # needed?
        },
    )
    return output["text"]

# Main app
def audiorec_demo_app():
    st.title("Speech-to-text (ASR) demo: distil-whisper-large")
    st.subheader('Press "Start Recording" / "Stop" to start/stop recording')

    wav_audio_data = st_audiorec()
    asr_text = ""
    if wav_audio_data is not None:
        asr_text = transcribe_speech(wav_audio_data)

    _ = st.text_input("Text:", value=asr_text)

if __name__ == '__main__':
    audiorec_demo_app()
