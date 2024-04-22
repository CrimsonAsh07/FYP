import threading
import cv2
import os
import time
import numpy as np
import requests
from Zero_DCE import lowlight_test_frame
from ultralytics import YOLO
from queue import Queue 
from threading import Lock  # Use a lock for thread synchronization
from threading import local

directions = ["none","left", "down", "right", "up"]
INPUT_MOB = 0
INPUT_WEBCAM = 1
INPUT_FILE = 2





def predict_direction(id,bbox, screen_dim):
    out_percent = 20/100
    if len(bbox) < 1:
        return 0

    #print("Len",id,bbox,screen_dim)
    if bbox[0] == 0 and bbox[2] < out_percent*screen_dim[1]:
        return 1   
    if bbox[1] == 0 and bbox[3] < out_percent*screen_dim[0]:
        return 4  
    if  (screen_dim[1]-bbox[2]) < out_percent*0.05*screen_dim[1] and bbox[0] > (1-out_percent)*screen_dim[1]:
        return 3 
    if (screen_dim[0]-bbox[3]) < out_percent*0.05*screen_dim[0] and bbox[1] > (1-out_percent)*screen_dim[0]:
        return 2
    
    return 0
def get_frame_from_url(url):
    img_stream = requests.get(url)
    img_arr = np.array(bytearray(img_stream.content), dtype=np.uint8)
    frame = cv2.imdecode(img_arr, -1)
    ret = frame is not None
    return ret, frame

def analyze_footage(inputType, inputPath,model,record_output,output_folder, duration,q, id, lock):
    if record_output and not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Initialize webcam
    if inputType > 0:    
        cap = cv2.VideoCapture(inputPath)
        if not cap.isOpened():
            print("Error: Unable to open input stream.")
            return

    start_time = time.time()
    count = 0

    while True:
        # Capture frame-by-frame
        if inputType > 0:
            ret, frame = cap.read()
        else:
            ret, frame = get_frame_from_url(inputPath)

        if not q.empty():
            data = q.queue[0]
            print("data1", data)
            with lock:
                if data["source"] != id:
                    data = q.get_nowait()
                    print("Got in",id,":",data)            
                    q.task_done()    

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

            results = model.track(enhanced_frame, persist=True, verbose=False, classes=[0])

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Display the annotated frame
            cv2.imshow("YOLOv8 Tracking"+str(id), annotated_frame)
            count+=1
            bboxs = results[0].boxes
            for bbox in bboxs:
                dir = predict_direction(bbox.id,bbox.xyxy.cpu().numpy()[0], bbox.orig_shape)
                if dir != 0 and bbox.id is not None: 
                    
                    
                    print("                                        ",str(id),bbox.id.item(),directions[dir])
                    q.put_nowait({"source": id, "destination": directions[dir], "person": bbox.id.item()})

            #print("Enhanced",enhanced_frame)
            #print("Frame",frame)
            #cv2.imwrite("./enhanced_frames/"+str(count)+".jpg", frame)
            #cv2.imwrite("./captured_frames/"+str(count)+".jpg", enhanced_frame)

        if duration is not None and time.time() - start_time > duration:  #to stop after duration seconds
            break
        
        # Esc key - exits application
        if cv2.waitKey(1) & 0xFF == 27:
            break
    if inputType > 0:  
        cap.release()


if __name__ == "__main__":
    record_output = False
    output_folder = "captured_frames"
    duration = None  # None - continuous recording || n - records for n seconds
    
    model1 = YOLO('yolov8n.pt')
    model2 = YOLO('yolov8n.pt')

    # Communicate using the Queue
    q = Queue() 
    lock = Lock()
    # Define the video files for the trackers
    path1 = "./test_data/cam_am.mp4"
    path2 = "./test_data/cam_bm.mp4"

    # Create the tracker threads
    tracker_thread1 = threading.Thread(target=analyze_footage, args=(INPUT_FILE,path2,model1,record_output,output_folder, duration,q,1,lock), daemon=True)
    tracker_thread2 = threading.Thread(target=analyze_footage, args=(INPUT_FILE,path1, model2,record_output,output_folder, duration,q,2, lock), daemon=True)

    # Start the tracker threads
    tracker_thread1.start()
    tracker_thread2.start()

    # Wait for the tracker threads to finish
    tracker_thread1.join()
    tracker_thread2.join()

    # Clean up and close windows
    cv2.destroyAllWindows()