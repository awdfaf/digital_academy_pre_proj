import streamlit as st
from main import Yolov7, renewal
import torch
import os
import argparse
from time import sleep
### crop 폴더 초기화를 위한 라이브러리
import shutil
### 미디어파이프를 위한 라이브러리
import math
import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import winsound as sd
### 멀티 프로세싱을 위한 라이브러리
import threading
import subprocess


# -------------------------------------------------------------------------------------------
def main():
    st.title("Dashboard")
    inference_msg = st.empty()
    st.sidebar.title("Configuration")
    input_source = st.sidebar.radio("Select input source", 
    ('RTSP/HTTPS', 'Webcam', 'Local video'))
    conf_thres = float(st.sidebar.text_input("Detection confidence", "0.50"))
    save_img = False
    
    
    if st.button("run"):
        print("aa")
        # subprocess.call("sum_copy7.py", shell=True)
        result = subprocess.run(['python', 'pre_project.py'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            print("외부 파일 실행 성공")
            print("표준 출력:\n", result.stdout)
        else:
            print("외부 파일 실행 실패")
            print("표준 에러 출력:\n", result.stderr)
    # -------------------------- WEBCAM ----------------------------------
    if input_source == "Webcam":
        if st.sidebar.button("Run"):
            stframe = st.empty()
            st.subheader("Inference Stats")
            if1, if2 = st.columns(2)
            st.subheader("System Stats")
            ss1, ss2, ss3 = st.columns(3)
            # Updating Inference results
            with if1:
                st.markdown("**Frame Rate**")
                if1_text = st.markdown("0")
            with if2:
                st.markdown("**Detected objects in current frame**")
                if2_text = st.markdown("0")
            # Updating System stats
            with ss1:
                st.markdown("**Memory Usage**")
                ss1_text = st.markdown("0")
            with ss2:
                st.markdown("**CPU Usage**")
                ss2_text = st.markdown("0")
            with ss3:
                st.markdown("**GPU Memory Usage**")
                ss3_text = st.markdown("0")
            # Run
            webcam_run = Yolov7(source='0', save_img=save_img, conf_thres=conf_thres,
                            stframe=stframe, if1_text=if1_text, if2_text=if2_text,
                            ss1_text=ss1_text, ss2_text=ss2_text, ss3_text=ss3_text)
            webcam_run.detect()
            
    
    
    
    
    torch.cuda.empty_cache()    
    
if __name__ == "__main__":
    try:
        ### 미디어파이프 전역변수 설정
        # Initializing mediapipe pose class.
        mp_pose = mp.solutions.pose
        # Setting up the Pose function.
        pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)
        # Initializing mediapipe drawing class, useful for annotation.
        mp_drawing = mp.solutions.drawing_utils 
        main()
    except SystemExit:
        pass
