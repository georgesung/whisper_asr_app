"""
Based on
https://github.com/whitphx/streamlit-webrtc/issues/357
"""
import os
import time

import streamlit as st
import torch
from aiortc.contrib.media import MediaRecorder
from streamlit_webrtc import ClientSettings, WebRtcMode, webrtc_streamer
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from utils.turn import get_ice_servers

# App settings
TEMP_AUDIO_FILE = "_temp.wav"
TEMP_TEXT_FILE = "_temp.txt"
INIT_KEY = "foo"

# App state
if INIT_KEY not in st.session_state:
    st.session_state[INIT_KEY] = "bar"
    with open(TEMP_TEXT_FILE, "w") as file:
        file.write("")

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
def transcribe_speech(filepath=TEMP_AUDIO_FILE):
    output = pipe(
        filepath,
        generate_kwargs={
            "task": "transcribe",
            "language": "english",  # needed?
        },
    )
    text = output["text"]
    with open(TEMP_TEXT_FILE, "w") as file:
        file.write(text)

# Streamlit app
st.title("Speech-to-text (ASR) demo: distil-whisper-large")

st.header("Press button to start/stop speaking")
def recorder_factory():
    return MediaRecorder(TEMP_AUDIO_FILE)

webrtc_ctx = webrtc_streamer(
    key="sendonly-audio",
    mode=WebRtcMode.SENDONLY,
    audio_receiver_size=256,
    rtc_configuration={"iceServers": get_ice_servers()},
    media_stream_constraints={"audio": True},
    in_recorder_factory=recorder_factory,
    on_audio_ended=transcribe_speech,
)

if not webrtc_ctx.state.playing:
    time.sleep(0.5)  # wait for the text file to be written by callback
    with open(TEMP_TEXT_FILE, "r") as file:
        text_input = file.read()
else:
    text_input = "listening..."
user_input = st.text_input("Text:", value=text_input)
