import streamlit as st
import pyaudio
import wave
import openai
import json

openai.api_key = "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"

st.title("Whisper")

st.write("Press the button and speak into your microphone to start whispering.")

if st.button("Start Recording"):
    st.write("Recording...")
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    RECORD_SECONDS = 5
    WAVE_OUTPUT_FILENAME = "output.wav"

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

    st.write("Recording finished. Click the button to start whispering.")

if st.button("Start Whispering"):
    st.write("Whispering...")
    with open("output.wav", "rb") as f:
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=f"Whisper: {json.dumps(f.read().hex())}",
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0.5,
        )

    text = response.choices[0].text.strip()

    st.write(text)
