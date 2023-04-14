import streamlit as st

# wav 파일 자동 업로드 함수 정의
@st.cache(allow_output_mutation=True)
def load_wav_file():
    # 파일 업로드
    uploaded_file = st.file_uploader("Upload a wav file", type="wav")
    if uploaded_file is not None:
        # 파일 내용 읽기
        wav_content = uploaded_file.read()
        # 업로드한 파일명 출력
        st.write("Uploaded wav file:", uploaded_file.name)
        return wav_content

# wav 파일 자동 업로드 함수 호출
wav_content = load_wav_file()

# 업로드한 파일의 내용 출력
if wav_content is not None:
    st.write("Wav file content:", wav_content)