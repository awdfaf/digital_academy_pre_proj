import cv2
import time
import torch
import argparse
import numpy as np
from torchvision import transforms
from utils.datasets import letterbox
from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.plots import output_to_keypoint
from utils.general import non_max_suppression_kpt, strip_optimizer
import os
import math

@torch.no_grad()

def run(
    poseweights='yolov7-w6-pose.pt',
    source='pose.mp4',
    device='cpu',
):
    video_path=source
    device = select_device(device)
    half = device.type!= 'cpu'
    model = attempt_load(poseweights, map_location=device)
    _ = model.eval()
    
    cap = cv2.VideoCapture(video_path)
    
    if (cap.isOpened() == False):
        print("Error")
        
    frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
    
    #code to write a video
    vid_write_image = letterbox(cap.read()[1], (frame_width), stride=64, auto=True)[0]
    resize_height, resize_width = vid_write_image.shape[:2]
    out_video_name = f"{video_path.split('/')[-1].split('.')[0]}"
    out = cv2.VideoWriter(f"{out_video_name}_test.mp4",
                        cv2.VideoWriter_fourcc(*'mp4v'), 30,
                        (resize_width, resize_height))

    #count no of frames
    frame_count = 0
    #count total fps
    total_fps = 0 


    #loop until cap opened or video not complete
    while(cap.isOpened):
        
        print("Frame {} Processing".format(frame_count))
        frame_count += 1  
        #get frame and success from video capture
        ret, frame = cap.read()
        #if success is true, means frame exist
        if ret:
            #store frame
            orig_image = frame

            #convert frame to RGB
            image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
            image = letterbox(image, (frame_width), stride=64, auto=True)[0]
            image = transforms.ToTensor()(image)
            image = torch.tensor(np.array([image.numpy()]))
            
            image = image.to(device)
            image = image.float()
            start_time = time.time()
            
            with torch.no_grad():
                output, _ = model(image)
            output = non_max_suppression_kpt(
                output, 0.5, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)
            output = output_to_keypoint(output)
            img = image[0].permute(1,2,0) * 255
            img = img.cpu().numpy().astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            

            

            # fall detection
            thre = (frame_height//2)+100
            for idx in range(output.shape[0]):
                #plot_skeleton_kpts(image, output[idx, 7:].T, 3)
                xmin, ymin = (output[idx, 2]-output[idx, 4]/2), (output[idx, 3]-output[idx, 5]/2)
                xmax, ymax = (output[idx, 2]+output[idx, 4]/2), (output[idx, 3]+output[idx, 5]/2)

                left_shoulder_y= output[idx][23]
                left_shoulder_x= output[idx][22]
                right_shoulder_y= output[idx][26]
                
                left_body_y = output[idx][41]
                left_body_x = output[idx][40]
                right_body_y = output[idx][44]

                len_factor = math.sqrt(((left_shoulder_y - left_body_y)**2 + (left_shoulder_x - left_body_x)**2 ))

                left_foot_y = output[idx][53]
                right_foot_y = output[idx][56]
                
                if left_shoulder_y > left_foot_y - len_factor and left_body_y > left_foot_y - (len_factor / 2) and left_shoulder_y > left_body_y - (len_factor / 2):
                    #Plotting key points on Image
                    cv2.rectangle(image,(int(xmin), int(ymin)),(int(xmax), int(ymax)),color=(0, 0, 255),
                        thickness=5,lineType=cv2.LINE_AA)
                    cv2.putText(image, 'Person Fell down', (11, 100), 0, 1, [0, 0, 2550], thickness=3, lineType=cv2.LINE_AA)
                    #bot.sendMessage(receiver_id, "Person Fall Detected")
                    filename = "C:/sfp/yolov7-main/out/a.jpg"
                    cv2.imwrite(filename, image)
                    #bot.sendPhoto(receiver_id, photo=open(filename, 'rb'))
                    os.remove(filename)
                
            
            
            # display preprocessed image
            img_ = img.copy()
            img_ = cv2.resize(
                img_, (960, 540), interpolation=cv2.INTER_LINEAR)
            cv2.imshow("Detection", img_)
            cv2.waitKey(1)
            # Calculate fps
            end_time = time.time()
            fps = 1 / (end_time - start_time)
            total_fps += fps
            frame_count += 1
            out.write(img)
        else:
            break
    cap.release()
    avg_fps = total_fps / frame_count
    print(f"Average FPS: {avg_fps:.3f}")





def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--poseweights', nargs='+', type=str,
                        default='yolov7-w6-pose.pt', help='model path(s)')
    parser.add_argument('--source', type=str, help='video/0 for webcam')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1')
    opt = parser.parse_args()
    return opt

def main(opt):
    run(**vars(opt))
    
    
    
    
    