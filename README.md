# Streamlit Whisper ASR app
Basic Streamlit app to record audio from user's microphone, run Automatic Speech Recognition (ASR) using distil-whisper-large, and show the transcribed text.

## Pre-reqs
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Set up HTTPS (if needed)
If running beyond localhost, you may need HTTPS so the browser can access your microphone.
Make note of the port the app is running on, e.g. 8501. Then set up an HTTPS proxy using https://github.com/suyashkumar/ssl-proxy/ (change the command below to match your OS):
```
./ssl-proxy-linux-amd64 -from 0.0.0.0:<external_https_port> -to 127.0.0.1:<internal_http_port>
```
E.g.
```
./ssl-proxy-linux-amd64 -from 0.0.0.0:8509 -to 127.0.0.1:8501
```
Where internal_http_port is the port the app is running on (e.g. 8501), external https port is the port to serve on (e.g. 8509 -> https://your_url:8509)

## How to run NiceGUI app
This app uses [NiceGUI](https://nicegui.io)
```
python ngui.py
```
![screenshot](/app_ngui_demo.png)

## How to run Streamlit app
This app uses https://github.com/stefanrmmr/streamlit-audio-recorder
```
streamlit run app.py
```
![screenshot](/app_demo.png)

## How to run Streamlit WebRTC app
This app uses https://github.com/whitphx/streamlit-webrtc
```
streamlit run app_webrtc.py
```
![screenshot](/app_webrtc_demo.png)
