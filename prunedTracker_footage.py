import cv2
import os
import time
import numpy as np
from Zero_DCE import lowlight_test_frame
from ultralytics import YOLO

modely = YOLO('yolov8n.pt', verbose=False)


directions = ["none","left", "down", "right", "up"]

def predict_direction(bbox, screen_dim):
    out_percent = 10/100
    if len(bbox) < 1:
        return 0
    bbox = bbox[0]
    #print("Len",bbox,screen_dim)
    if bbox[0] == 0 and bbox[2] < out_percent*screen_dim[1]:
        return 1   
    if bbox[1] == 0 and bbox[3] < out_percent*screen_dim[0]:
        return 4  
    if  (screen_dim[1]-bbox[2]) < out_percent*0.05*screen_dim[1] and bbox[0] > (1-out_percent)*screen_dim[1]:
        return 3 
    if (screen_dim[0]-bbox[3]) < out_percent*0.05*screen_dim[0] and bbox[1] > (1-out_percent)*screen_dim[0]:
        return 2
    
    return 0


def analyze_footage(video_path, output_folder, duration=None):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    #Initialize Video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Unable to open video.")
        return

    capture_count = 1
    start_time = time.time()
    count = 0

    while True:
        ret, frame = cap.read()

        if ret:
            if duration is not None and time.time() - start_time > duration:
                break

            enhanced_frame = lowlight_test_frame.lowlight(frame, threshold=75, verbose=False)  # low light threshold
            results = modely.track(enhanced_frame, persist=True, verbose=False, classes=[0])
            bbox = results[0].boxes

            annotated_frame = results[0].plot()
            bbox = results[0].boxes
            dir = predict_direction(bbox.xyxy.cpu().numpy(), bbox.orig_shape)
            if(dir != 0): 
                print(directions[dir])
            cv2.imshow("YOLOv8 Tracking", annotated_frame)
            cv2.namedWindow("YOLOv8 Tracking", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("YOLOv8 Tracking", 1024,1024)  

            count += 1

            if duration is None:
                cv2.waitKey(1)  # Display frame indefinitely if duration is None
            else:
                if cv2.waitKey(1) & 0xFF == 27:
                    break

        else:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    footage_path = "test_data/cam_a.mp4"
    output_folder = "captured_frames"
    duration = None  # None - continuous processing || n - process for n seconds

    analyze_footage(footage_path, output_folder, duration)
