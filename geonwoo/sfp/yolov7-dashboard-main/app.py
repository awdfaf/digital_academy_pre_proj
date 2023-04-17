import streamlit as st
from main import Yolov7
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



        

    
    
# -------------------------------------------------------------------------------------------

            

def renewal():
    crop_path = 'crop'
    
    while True:
        sleep(3)
        
        if os.path.exists(crop_path):
            try:
                
                list = os.listdir(crop_path)
                for index in range(len(list)):
                    remove_path = crop_path + '/' + list[index]    
                    os.remove(remove_path)
                
            except PermissionError:
                print("Permission denied: Unable")
            except Exception as e:
                print("Error :",e)
        else:
            print("Folder not found")
    


def calculateAngle(landmark1, landmark2, landmark3):

    # Get the required landmarks coordinates.
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3

    # Calculate the angle between the three points
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    
    # Check if the angle is less than zero.
    if angle < 0:

        # Add 360 to the found angle.
        angle += 360
    
    # Return the calculated angle.
    return angle


def detectPose(image, pose, display=True):
    
    # Create a copy of the input image.
    output_image = image.copy()
    
    # Convert the image from BGR into RGB format.
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Perform the Pose Detection.
    results = pose.process(imageRGB)
    
    # Retrieve the height and width of the input image.
    height, width, _ = image.shape
    
    # Initialize a list to store the detected landmarks.
    landmarks = []
    
    # Check if any landmarks are detected.
    if results.pose_landmarks:
    
        # Draw Pose landmarks on the output image.
        mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,
                                connections=mp_pose.POSE_CONNECTIONS)
        
        # Iterate over the detected landmarks.
        for landmark in results.pose_landmarks.landmark:
            
            # Append the landmark into the list.
            landmarks.append((int(landmark.x * width), int(landmark.y * height),
                                  (landmark.z * width)))
    
    # Check if the original input image and the resultant image are specified to be displayed.
    if display:
    
        # Display the original input image and the resultant image.
        plt.figure(figsize=[22,22])
        plt.subplot(121);plt.imshow(image[:,:,::-1]);plt.title("Original Image");plt.axis('off');
        plt.subplot(122);plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
        
        # Also Plot the Pose landmarks in 3D.
        mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
        
    # Otherwise
    else:
        
        # Return the output image and the found landmarks.
        return output_image, landmarks


def classifyPose(landmarks, output_image, display=False):
    
    # Initialize the label of the pose. It is not known at this stage.
    label = 'Unknown Pose'

    # Specify the color (Red) with which the label will be written on the image.
    color = (0, 0, 255)
    
    # Calculate the required angles.
    #----------------------------------------------------------------------------------------------------------------
    
    # Get the angle between the left shoulder, elbow and wrist points. 
    left_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])
    
    # Get the angle between the right shoulder, elbow and wrist points. 
    right_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])   
    
    # Get the angle between the left elbow, shoulder and hip points. 
    left_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])

    # Get the angle between the right hip, shoulder and elbow points. 
    right_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])

    # Get the angle between the left hip, knee and ankle points. 
    left_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])

    # Get the angle between the right hip, knee and ankle points 
    right_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
    
    print("left_elbow_angle = {}, right_elbow_angle = {}".format(left_elbow_angle, right_elbow_angle))
    print("left_shoulder_angle = {}, right_shoulder_angle = {}".format(left_shoulder_angle, right_shoulder_angle))
    print("left_knee_angle = {}, right_knee_angle = {}".format(left_knee_angle, right_knee_angle))
    
    ### 포즈 라벨링 처리            
    #----------------------------------------------------------------------------------------------------------------
    
      
    #--------------------------------
    
    
    # Check if the pose is classified successfully
    if label != 'Unknown Pose':
        
        print("설정된 포즈가 없거나, 탐지할 수 없습니다.")
        # Update the color (to green) with which the label will be written on the image.
        color = (0, 255, 0)  

    # Write the label on the output image. 
    cv2.putText(output_image, label, (10, 30),cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
    
    # Check if the resultant image is specified to be displayed.
    if display:
    
        # Display the resultant image.
        plt.figure(figsize=[10,10])
        plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
        
    else:
        
        # Return the output image and the classified label.
        return output_image, label

       
### 미디어파이프 함수
### - 멀티프로세싱에 사용된다.
def mediapipe():
    directory = "crop/" # 폴더 경로 입력

    while True:
        sleep(3)
        try :
            print("미디어파이프 폴더 안 탐색 ",os.listdir(directory))
            image = cv2.imread('./crop/{}'.format(os.listdir(directory)[0]))
            sleep(1)
            output_image, landmarks = detectPose(image,mp_pose.Pose(static_image_mode=True,
                                                        min_detection_confidence=0.5, model_complexity=0),display=False)
            sleep(2)
        
            print("미디어파이프 접근 성공")    
            sleep(2)
            if landmarks:
                print("랜드마크에 접근")
                classifyPose(landmarks, output_image, display=True)
        except :
            sleep(2)
            continue
 
    
    # -------------------------------------------------------------------------------------------
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
            
            # thread_yolov7 = threading.Thread(target=webcam_run.detect)
            # thread_renewal = threading.Thread(target=renewal)
            # thread_mediapipe = threading.Thread(target=mediapipe)

            # thread_renewal.start()
            # thread_mediapipe.start()
            # sleep(3)
            # thread_yolov7.start()

            # thread_yolov7.join()
            # thread_renewal.join()
            # thread_mediapipe.join()
            
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
