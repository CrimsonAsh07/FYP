import cv2
import os
import time
import numpy as np
from Zero_DCE import lowlight_test_frame
from ultralytics import YOLO

modely = YOLO('yolov8n.pt')

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

            enhanced_frame = lowlight_test_frame.lowlight(frame, threshold=75)  # low light threshold
            # enhanced_frame = enhanced_frame * 255
            # enhanced_frame = enhanced_frame.astype(np.uint8)
            results = modely.track(enhanced_frame, persist=True)

            annotated_frame = results[0].plot()

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
