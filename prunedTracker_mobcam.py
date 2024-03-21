# Import essential libraries
import cv2
import os
import time
import requests
import numpy as np
from Zero_DCE import lowlight_test_frame
from ultralytics import YOLO

modely = YOLO('yolov8n.pt')

directions = ["none", "left", "down", "right", "up"]

def predict_direction(id, bbox, screen_dim):
    out_percent = 20 / 100
    if len(bbox) < 1:
        return 0

    if bbox[0] == 0 and bbox[2] < out_percent * screen_dim[1]:
        return 1
    if bbox[1] == 0 and bbox[3] < out_percent * screen_dim[0]:
        return 4
    if (screen_dim[1] - bbox[2]) < out_percent * 0.05 * screen_dim[1] and bbox[0] > (1 - out_percent) * screen_dim[1]:
        return 3
    if (screen_dim[0] - bbox[3]) < out_percent * 0.05 * screen_dim[0] and bbox[1] > (1 - out_percent) * screen_dim[0]:
        return 2

    return 0

def get_frame_from_url(url):
    img_stream = requests.get(url)
    img_arr = np.array(bytearray(img_stream.content), dtype=np.uint8)
    frame = cv2.imdecode(img_arr, -1)
    return frame

def analyze_footage(record_output, output_folder, duration, url):
    if record_output and not os.path.exists(output_folder):
        os.makedirs(output_folder)

    capture_count = 1
    start_time = time.time()
    count = 0

    while True:
        frame = get_frame_from_url(url)

        if frame is not None:

            if record_output and time.time() - start_time > 1:
                filename = os.path.join(output_folder, f"{capture_count}.jpg")
                cv2.imwrite(filename, frame)

                start_time = time.time()
                capture_count += 1

            enhanced_frame = lowlight_test_frame.lowlight(frame, threshold=50, verbose=False)
            results = modely.track(enhanced_frame, persist=True, verbose=False, classes=[0])
            annotated_frame = results[0].plot()

            resized_frame = cv2.resize(annotated_frame, (960, 540))
            cv2.imshow(str(url), resized_frame)
            
            count += 1
            bboxs = results[0].boxes
            for bbox in bboxs:
                dir = predict_direction(bbox.id, bbox.xyxy.cpu().numpy()[0], bbox.orig_shape)
                if dir != 0:
                    print(bbox.id, directions[dir])

        if duration is not None and time.time() - start_time > duration:
            break

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    record_output = False
    output_folder = "captured_frames"
    duration = None
    
    camA = "http://192.168.29.24:8080/shot.jpg"
    camB = "http://192.168.29.24:8080/shot.jpg"  

    analyze_footage(record_output, output_folder, duration, camA)

