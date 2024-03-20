import cv2
import os
import time

from Zero_DCE import lowlight_test_frame
from ultralytics import YOLO

modely = YOLO('yolov8n.pt')

directions = ["none","left", "down", "right", "up"]

def predict_direction(bbox, screen_dim):
    out_percent = 20/100
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

def analyze_footage(record_output,output_folder, duration):
    if record_output and not os.path.exists(output_folder):
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
    count = 0

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
     
        if ret:

            # Save one frame per second
            if record_output and  time.time() - start_time > 1:

                filename = os.path.join(output_folder, f"{capture_count}.jpg")
                cv2.imwrite(filename, frame)
                
                # Reset timer
                start_time = time.time()

                capture_count += 1
                
            # Display webcam feed
            #cv2.imshow('Webcam', frame)

            enhanced_frame= lowlight_test_frame.lowlight(frame, threshold=50,verbose=False) #low light threshold

            results = modely.track(enhanced_frame, persist=True, verbose=False, classes=[0])

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Display the annotated frame
            cv2.imshow("YOLOv8 Tracking", annotated_frame)
            count+=1
            bbox = results[0].boxes
            dir = predict_direction(bbox.xyxy.cpu().numpy(), bbox.orig_shape)
            if(dir != 0): 
                print(directions[dir])

            #print("Enhanced",enhanced_frame)
            #print("Frame",frame)
            #cv2.imwrite("./enhanced_frames/"+str(count)+".jpg", frame)
            #cv2.imwrite("./captured_frames/"+str(count)+".jpg", enhanced_frame)

        if duration is not None and time.time() - start_time > duration:  #to stop after duration seconds
            break
        
        # Esc key - exits application
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    record_output = False
    output_folder = "captured_frames"
    duration = None  # None - continuous recording || n - records for n seconds
    
    analyze_footage(record_output,output_folder, duration)
