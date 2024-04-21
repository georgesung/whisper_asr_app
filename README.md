# Streamlit Whisper ASR app
Basic Streamlit app to record audio from user's microphone, run Automatic Speech Recognition (ASR) using distil-whisper-large, and show the transcribed text.

![screenshot](/app_demo.png)


## Pre-reqs
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## How to run
```
streamlit run st_app.py
```

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
