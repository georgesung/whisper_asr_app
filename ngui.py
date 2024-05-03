import base64
from typing import Callable, Optional

import torch
from nicegui import app, events, ui
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


"""
Record audio and transcribe speech, using NiceGUI front-end
For front-end code, see audio_recorder.vue
"""

### ASR model & pipeline ###
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

def transcribe_speech(audio):
    output = pipe(
        audio,
        generate_kwargs={
            "task": "transcribe",
            "language": "english",  # needed?
        },
    )
    return output["text"]


### Audio recorder helper class ###
class AudioRecorder(ui.element, component='audio_recorder.vue'):

    def __init__(self, *, on_audio_ready: Optional[Callable] = None) -> None:
        super().__init__()
        self.recording = b''

        def handle_audio(e: events.GenericEventArguments) -> None:
            self.recording = base64.b64decode(e.args['audioBlobBase64'].encode())
            if on_audio_ready:
                on_audio_ready(self.recording)
        self.on('audio_ready', handle_audio)

    def start_recording(self) -> None:
        self.run_method('startRecording')

    def stop_recording(self) -> None:
        self.run_method('stopRecording')

    def play_recorded_audio(self) -> None:
        self.run_method('playRecordedAudio')


### NiceGUI app ###
@ui.page('/')
def index():
    with ui.row().classes('w-full justify-center'):
        app.storage.client['audio_recorder'] = \
            AudioRecorder(on_audio_ready=lambda audio: result_label.set_text(transcribe_speech(audio)))
        audio_recorder = app.storage.client.get('audio_recorder')

    with ui.row().classes('w-full justify-center'):
        ui.button('Play', on_click=audio_recorder.play_recorded_audio) \
            .bind_enabled_from(audio_recorder, 'recording')
        ui.button('Download', on_click=lambda: ui.download(audio_recorder.recording, 'audio.ogx')) \
            .bind_enabled_from(audio_recorder, 'recording')

    with ui.row().classes('w-full justify-center'):
        result_label = ui.label('Recognition result will appear here')

ui.run(port=8508)
