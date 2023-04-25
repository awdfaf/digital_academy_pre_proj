import streamlit as st
import os
from pydub import AudioSegment
from io import BytesIO
import time
from scipy.io import wavfile
from scipy.signal import butter, lfilter
import numpy as np
import whisper

import requests
import json

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
# nltk.download('punkt')

model = whisper.load_model("medium")

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
high_freq = 3000

# 발급받은 클라이언트 ID와 클라이언트 시크릿을 입력하세요.
client_id = "8QNC6WEut1cZ5r1Flsc6"
client_secret = "ci6mvyty4v"
target_lang = "ko"

def detect_language(text):
    url = "https://openapi.naver.com/v1/papago/detectLangs"
    headers = {
        "X-Naver-Client-Id": client_id,
        "X-Naver-Client-Secret": client_secret,
    }
    data = {"query": text}
    response = requests.post(url, headers=headers, data=data)
    result = json.loads(response.text)
    st.write(result["langCode"])
    return result["langCode"]

def translate_text(text, target_lang):
    source_lang = detect_language(text)
    if source_lang == "ko" :
        return text 
    else:
        url = "https://openapi.naver.com/v1/papago/n2mt"
        headers = {
            "X-Naver-Client-Id": client_id,
            "X-Naver-Client-Secret": client_secret,
        }
        data = {
            "source": source_lang,
            "target": target_lang,
            "text": text,
        }
        response = requests.post(url, headers=headers, data=data)
        result = json.loads(response.text)
        translated_text = result["message"]["result"]["translatedText"]
        return translated_text

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
    sound = AudioSegment.from_file("./wav파일들2/output_file.wav", format="wav")
    louder_sound = sound + 50
    louder_sound.export("./wav파일들2/output_file.wav", format="wav")
    result = model.transcribe("./wav파일들2/output_file.wav")
    # st.write(result["text"])
    text = result["text"]
    translated_text = translate_text(text, target_lang)
    st.write(translated_text)
    
    tokens = nltk.word_tokenize(translated_text)
    tokens = [" ".join(tokens)]
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(tokens)
    lda_model = LatentDirichletAllocation(n_components=1, random_state=42)
    lda_model.fit(X)
    feature_names = vectorizer.get_feature_names_out()
    top_words = lda_model.components_[0].argsort()[:-11:-1]
    topic_words = [feature_names[i] for i in top_words]
    st.write("추출된 Topic:", ", ".join(topic_words))
    
time.sleep(10)

while True :
    files2 = os.listdir(path)
    wav_files = [f for f in files2 if f.endswith('.wav')]
    st.write('WAV files in folder:', wav_files[1:])
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
                filtered_data = lfilter(b, a, data) + 50
                wavfile.write("./wav파일들2/output_file.wav", rate, np.asarray(filtered_data, dtype=np.int16))
                audio_file = AudioSegment.from_file(os.path.join(path, "output_file.wav"))
                audio_file.export(buffer, format='wav')
                st.audio(buffer.getvalue(), format='audio/wav')
                sound = AudioSegment.from_file("./wav파일들2/output_file.wav", format="wav")
                louder_sound = sound + 55
                louder_sound.export("./wav파일들2/output_file.wav", format="wav")
                result = model.transcribe("./wav파일들2/output_file.wav")
                # st.write(result["text"]) 
                text = result["text"]
                translated_text = translate_text(text, target_lang)
                st.write(translated_text)
                tokens = nltk.word_tokenize(translated_text)
                tokens = [" ".join(tokens)]
                vectorizer = CountVectorizer(stop_words='english')
                X = vectorizer.fit_transform(tokens)
                lda_model = LatentDirichletAllocation(n_components=1, random_state=42)
                lda_model.fit(X)
                feature_names = vectorizer.get_feature_names_out()
                top_words = lda_model.components_[0].argsort()[:-11:-1]
                topic_words = [feature_names[i] for i in top_words]
                st.write("추출된 Topic:", ", ".join(topic_words))
        files1 = files2
        
    time.sleep(10)