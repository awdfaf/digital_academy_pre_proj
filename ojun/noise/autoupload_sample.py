import streamlit as st
import os
from pydub import AudioSegment
from io import BytesIO
import time
from scipy.io import wavfile
from scipy.signal import butter, lfilter
import numpy as np
import whisper

model = whisper.load_model("tiny")

# wav 파일이 저장된 폴더 경로
path = './wav파일들2'

# 폴더 내의 모든 파일 목록 가져오기
files1 = os.listdir(path)

# wav 파일만 선택하여 처리
wav_files = [f for f in files1 if f.endswith('.wav')]

# wav 파일 목록 출력
st.write('WAV files in folder:', wav_files)

### bandpassfilter 설정
low_freq = 100
high_freq = 1000

# 선택한 wav 파일 로드 및 재생
for selected_file in wav_files:
    st.write('Playing:', selected_file)
    audio_file = AudioSegment.from_file(os.path.join(path, selected_file))
    buffer = BytesIO()
    audio_file.export(buffer, format='wav')
    
    # wav 파일 로드
    rate, data = wavfile.read(buffer)

    # 필터링을 위한 상수 설정
    nyq = 0.5 * rate
    low = low_freq / nyq
    high = high_freq / nyq
    order = 5

    # Butterworth bandpass 필터 계수 계산
    b, a = butter(order, [low, high], btype='band')

    # 필터 적용
    filtered_data = lfilter(b, a, data)
    
    wavfile.write("./wav파일들2/output_file.wav", rate, np.asarray(filtered_data, dtype=np.int16))
    
    # wav_files = [f for f in files1 if f.endswith('output_file.wav')]
        
    audio_file.export(buffer, format='wav')
    st.audio(buffer.getvalue(), format='audio/wav')
    result = model.transcribe("./wav파일들2/output_file.wav")
    st.write(result["text"])
    files1 = os.listdir(path)
    
time.sleep(10)

while True :
    files2 = os.listdir(path)
    wav_files = [f for f in files2 if f.endswith('.wav')]
    st.write('WAV files in folder:', wav_files)
    if files2 != files1 :
        for selected_file in files2 :
            if selected_file not in files1 :
                if selected_file == "output_file.wav":
                    pass 
                st.write('Playing:', selected_file)
                audio_file = AudioSegment.from_file(os.path.join(path, selected_file))
                buffer = BytesIO()
                audio_file.export(buffer, format='wav')
                rate, data = wavfile.read(buffer)
                nyq = 0.5 * rate
                low = low_freq / nyq
                high = high_freq / nyq
                order = 5
                b, a = butter(order, [low, high], btype='band')
                filtered_data = lfilter(b, a, data)
                wavfile.write("./wav파일들2/output_file.wav", rate, np.asarray(filtered_data, dtype=np.int16))
                audio_file = AudioSegment.from_file(os.path.join(path, "output_file.wav"))
                audio_file.export(buffer, format='wav')
                st.audio(buffer.getvalue(), format='audio/wav')
                result = model.transcribe("./wav파일들2/output_file.wav")
                st.write(result["text"]) 
        files1 = files2
    
    time.sleep(10)