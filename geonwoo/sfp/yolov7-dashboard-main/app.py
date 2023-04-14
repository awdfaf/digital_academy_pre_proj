import streamlit as st
from main import Yolov7
import torch
import os
import argparse
import mediapipe as mp

def main():

    st.title("Dashboard")
    inference_msg = st.empty()
    st.sidebar.title("Configuration")

    input_source = st.sidebar.radio("Select input source", 
    ('RTSP/HTTPS', 'Webcam', 'Local video'))
    
    conf_thres = float(st.sidebar.text_input("Detection confidence", "0.50"))

    save_img = False
    
    # ------------------------- LOCAL VIDEO ------------------------------
    if input_source == "Local video":

        video = st.sidebar.file_uploader("Select input video", type=["mp4", "avi"], accept_multiple_files=False)

        # save video temporarily to process it using cv2
        if video is not None:
            if not os.path.exists('./tempDir'):
                os.makedirs('./tempDir')
            with open(os.path.join(os.getcwd(), "tempDir", video.name), "wb") as file:
                file.write(video.getbuffer())
            
            video_filename = f'./tempDir/{video.name}'
        

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
            local_run = Yolov7(source=video_filename, save_img=save_img, conf_thres=conf_thres,
                            stframe=stframe, if1_text=if1_text, if2_text=if2_text,
                            ss1_text=ss1_text, ss2_text=ss2_text, ss3_text=ss3_text)

            local_run.detect()
            inference_msg.success("Inference Complete!")

            # delete the saved video
            if os.path.exists(video_filename):
                os.remove(video_filename)

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
        
    # -------------------------- RTSP/HTTPS ------------------------------
    if input_source == "RTSP/HTTPS":
        
        rtsp_input = st.sidebar.text_input("Video link", "https://www.youtube.com/watch?v=zu6yUYEERwA")

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
            stream_run = Yolov7(source=rtsp_input, save_img=save_img, conf_thres=conf_thres,
                            stframe=stframe, if1_text=if1_text, if2_text=if2_text,
                            ss1_text=ss1_text, ss2_text=ss2_text, ss3_text=ss3_text)
                            
            stream_run.detect()

    torch.cuda.empty_cache()

if __name__ == "__main__":
    try:
        ### 전역변수 지정
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
        parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
        parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
        parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
        parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--view-img', action='store_true', help='display results')
        parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
        parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
        parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--update', action='store_true', help='update all models')
        parser.add_argument('--project', default='runs/detect', help='save results to project/name')
        parser.add_argument('--name', default='exp', help='save results to project/name')
        parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
        parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
        opt = parser.parse_args()
        print("opt ==> ", opt)
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
