import streamlit as st
import pyaudio
import wave
import openai
import json
from config import get_page_config, get_whisper_settings, save_whisper_settings
from core import MediaManager
import os
import pyaudio
import wave
from scipy.io import wavfile
from scipy.signal import butter, lfilter
import numpy as np
import matplotlib.pyplot as plt
import whisper

st.set_page_config(**get_page_config())


# Session states
# ----------------------------------------------------------------
# Set session state to toggle list & detail view
if "list_mode" not in st.session_state:
    st.session_state.list_mode = True
    st.session_state.selected_media = None
    st.session_state.selected_media_offset = 0

# Add whisper settings to session state
if "whisper_params" not in st.session_state:
    st.session_state.whisper_params = get_whisper_settings()

if "media_manager" not in st.session_state:
    st.session_state.media_manager = MediaManager()

# Alias for session state media manager
media_manager = st.session_state.media_manager
# ----------------------------------------------------------------


# 녹음기능 
# ----------------------------------------------------------------
if st.button("Start Recording"):
    st.write("Recording...")
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    RECORD_SECONDS = 5
    WAVE_OUTPUT_FILENAME = "./output/output.wav"

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    
# ----------------------------------------------------------------
    
# 음성 파일 불러오기
# ----------------------------------------------------------------
    # Update session state whisper params
    st.session_state.whisper_params["task"] = "transcribe"
    source_type = "upload"  
    file_path = "./output/output.wav"
    if file_path:
        try:
            with open(file_path, "rb") as f:
                source = f.read()
                # 파일 내용에 대한 처리를 수행합니다.
                st.write("녹음 완료")
        except:
            st.write("올바른 경로를 입력해주세요.")


    
# whisper
# ----------------------------------------------------------------

    model = whisper.load_model("medium")
    result = model.transcribe(source)
    st.write(result["text"])











