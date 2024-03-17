import cv2
import os
import time

def record_webcam(output_folder, duration):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to open webcam.")
        return
    
    # Initialize variables
    existing_files_count = len(os.listdir(output_folder))

    capture_count = existing_files_count+1
    start_time = time.time()
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if ret:

            # Save one frame per second
            if time.time() - start_time > 1:

                filename = os.path.join(output_folder, f"{capture_count}.jpg")
                cv2.imwrite(filename, frame)
                
                # Reset timer
                start_time = time.time()

                capture_count += 1
                
            # Display webcam feed
            cv2.imshow('Webcam', frame)
        
        if duration is not None and time.time() - start_time > duration:
            break
        
        # Esc key - exits application
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    output_folder = "captured_frames"
    duration = None  # None - continuous recording || n - records for n seconds
    
    record_webcam(output_folder, duration)
