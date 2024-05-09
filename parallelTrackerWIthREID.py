import threading
import cv2
import os
import time
import numpy as np
import requests

from Zero_DCE import lowlight_test_frame
from ultralytics import YOLO
from threading import Lock  # Use a lock for thread synchronization
from mapper.get_topology_query import *
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from customPlotter import cplot
#REID MODEL
from parse_config import ConfigParser
from getModel import getModelFromConfig, get_pairing
import argparse
import torch
from event_logger import sqlite
from event_logger import event_logger

directions = ["none","left", "down", "right", "up"]
INPUT_MOB = 0
INPUT_WEBCAM = 1
INPUT_FILE = 2

STATUS_IDD = 1
STATUS_PENDING = 2

TENSOR_EDGE = 0
TENSOR_MID = 1

def show_image(image_tensor):
    output_tensor = image_tensor.squeeze(0).cpu().detach().permute(1, 2, 0)
     # Permute to (height, width, channels)

    plt.imshow(output_tensor)  # Remove batch dimension, convert to CPU, detach
    plt.axis('off')  # Hide axes
    plt.show()

class Person: ## object of global shared_person
    def __init__(self, id, camera_id, photo):
        self.status = STATUS_IDD
        self.id = id
        self.camera_id = camera_id
        self.photo = None

class localMappingPerson: #object of local_id_mapping 
    def __init__(self,g_id,status, entry):
     
        self.status = status
        self.g_id = g_id
        self.entry = entry
        self.frames = 0

class ReIDPerson: #reid_map queue object
    def __init__(self, g_id, camera_source):
     
        self.status = STATUS_IDD
        self.g_id = g_id
        self.camera_source = camera_source
    

    def __str__(self):
        return f"status: {self.status}, g_id: {self.g_id}, cam: {self.camera_source}"    

class LockArray:
    def __init__(self):
        self.lock = Lock()
        self.q = {}

# class toReID:
#     def __init__(self, pid, photo):
#         self.pid = pid
#         self.photo = photo


def update_person_location(person_id, camera_id):
    with shared_person_lock:
        shared_person[person_id].camera_id = camera_id

def add_person(person):
    with shared_person_lock:
        while person.id in shared_person:
            person.id +=1 
        shared_person[person.id] = person
    return person.id

def update_person_image(person_id, image):
    with shared_person_lock:
        if person_id in shared_person:
            shared_person[person_id].photo = image




def predict_direction(id,bbox, screen_dim):
    out_percent = 27/100
    if len(bbox) < 1:
        return 0

    
    if bbox[0] < out_percent*0.5*screen_dim[1] and bbox[2] < out_percent*screen_dim[1]:
        return 1   
    if bbox[1] < out_percent*0.5*screen_dim[0] and bbox[3] < out_percent*screen_dim[0]:
        return 4  
    if  (screen_dim[1]-bbox[2]) < out_percent*0.5*screen_dim[1] and bbox[0] > (1-out_percent)*screen_dim[1]:
        return 3 
    if (screen_dim[0]-bbox[3]) < out_percent*0.5*screen_dim[0] and bbox[1] > (1-out_percent)*screen_dim[0]:
        return 2
    #print(bbox[0], bbox[1], bbox[2], bbox[3],end = " :: ")
    
    return 0

def get_frame_from_url(url):
    img_stream = requests.get(url)
    img_arr = np.array(bytearray(img_stream.content), dtype=np.uint8)
    frame = cv2.imdecode(img_arr, -1)
    ret = frame is not None
    return ret, frame

def analyze_footage(inputType, inputPath,record_output,output_folder, duration, id, reidModel):
   
    model = YOLO('yolov8n.pt')
    if record_output and not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Initialize webcam and file cam
    if inputType > 0:    
        cap = cv2.VideoCapture(inputPath)
        

        if not cap.isOpened():
            print("Error: Unable to open input stream.")
            return
        if inputType == INPUT_WEBCAM:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 7680)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 4320)
    start_time = time.time()
    count = 0

    local_id_mapping = {}
    restricted_detections = {}
    

# Print the current resolution
    
    conn = sqlite.sqlite3.connect('surveillance_db.db') # connect to sqlite
    print("\n\n\n")
    while True:
        # Capture frame-by-frame
        if inputType > 0:
            ret, frame = cap.read()
        else:
            ret, frame = get_frame_from_url(inputPath)

                #[print(x) for x in reid_map[id].q.values()]           
                 

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

            results = model.track(enhanced_frame, persist=True, verbose=False, classes=[0], conf=0.3)

            # Visualize the results on the frame
            
            

            count+=1
            bboxs = results[0].boxes
            to_be_reid = {}
            newIds=[]
            local_id_mapping_copy = local_id_mapping.copy()

            for bbox in bboxs:

                dir = predict_direction(bbox.id,bbox.xyxy.cpu().numpy()[0], bbox.orig_shape)
                if bbox.id is not None:
                    
                    bboxId = bbox.id.item()
                    if bboxId in local_id_mapping_copy:
                        local_id_mapping_copy.pop(bboxId)
                        

                    if bbox.conf.item() > 0.7 and bboxId not in restricted_detections:
                        if node_isRestricted(graph,chr(id + ord('0'))) == "Unrestricted Area":
                            pass
                        else:
                            restricted_detections[bboxId]='1'

                if dir == 0 and bbox.id is not None:
                    if bbox.conf.item() > 0.7:
                        boxes = bbox.xyxy.tolist()[0]
                        crop_object = frame[int(boxes[1]):int(boxes[3]), int(boxes[0]):int(boxes[2])]
                        if bboxId not in local_id_mapping and bboxId not in to_be_reid:
                            print(id,f"CAMERA {id} :: Person {bboxId} has been located at the center.")
                            pid = add_person(Person(bboxId,id, crop_object))
                            cv2.imwrite(log_folder_path+f'/{bboxId}.jpg', crop_object)
                            event_logger.write_log(log_file_path, f"Person {bboxId} has been located at camera {id}")
                            sqlite.record_entry(id,bboxId, conn)
                            local_id_mapping[bboxId] = localMappingPerson(pid, STATUS_IDD, dir)

                        if bboxId  in local_id_mapping:
                            local_id_mapping[bboxId].entry = dir

                if dir != 0 and bbox.id is not None: 
                    
                    bboxId = bbox.id.item()
                    #if someone new detected, find out if he needs to be reid
                        
                    #if re id then reid, add to local map, and update global map
                    #if truly new then add to local map and add to global map

                    if bbox.conf.item() > 0.5:
                        #if id == 0: print(id,directions[dir], end=" ")
                        boxes = bbox.xyxy.tolist()[0]
                        crop_object = frame[int(boxes[1]):int(boxes[3]), int(boxes[0]):int(boxes[2])]
                        
                        if bboxId not in local_id_mapping and bboxId not in to_be_reid:
                            
                            if len(reid_map[id].q) > 0 and bboxId not in to_be_reid: #add to reid queue
                                print(f"CAMERA {id} :: (To Be ReIdentified) Person L{bboxId} coming from",directions[dir])
                                to_be_reid[bboxId] = (crop_object,dir)
                                
                                #local_id_mapping[bbox.id] = localMappingPerson(-1,STATUS_PENDING, ) 
            
                            elif len(reid_map[id].q)== 0: #new person
                                print(f"CAMERA {id} :: Person {bboxId} detected at the", directions[dir], "section of the room.")
                                pid = add_person(Person(bboxId,id, crop_object))
                                cv2.imwrite(log_folder_path+f'/{bboxId}.jpg', crop_object)
                                event_logger.write_log(log_file_path, f"Person {bboxId} has been located at camera {id}")
                                sqlite.record_entry(id,bboxId, conn)
                                local_id_mapping[bboxId] = localMappingPerson(pid, STATUS_IDD, dir)
                        else:
                            get_dest = query_graph(chr(id + ord('0')), directions[dir], graph)
                            g_id = local_id_mapping[bboxId].g_id
                            if(local_id_mapping[bboxId].entry != dir) and get_dest is not None:
                                
                                if g_id not in reid_map[int(get_dest)].q:
                                    with reid_map[int(get_dest)].lock:
                                        print(f"CAMERA {id} :: Predicted direction of motion:", directions[dir],end = " ")
                                        print(f"|| Predicted Camera to reappear: ",get_dest)
                                        reid_map[int(get_dest)].q[g_id] = ReIDPerson(g_id, id)
                                        update_person_image(g_id, crop_object)
                                        cv2.imwrite(log_folder_path+f'/{g_id}.jpg', crop_object)


                #mapping ids to global
                                
                if(bbox.id is not None):
                    idItem = bbox.id.item()
                    if (idItem in local_id_mapping):
                        newIds.append(local_id_mapping[idItem].g_id)
                    else:
                        newIds.append(idItem)

            #Remove people who left
            for ids in local_id_mapping_copy:
                if ids in local_id_mapping:
                    if local_id_mapping[ids].frames > 30:
                        local_id_mapping.pop(ids)        
                    else: 
                        local_id_mapping[ids].frames+=1
                    

            
            new_results = results[0].new()
            annotated_frame= cplot(new_results, newIds, results[0].boxes)
            annotated_frame = cv2.resize(annotated_frame, (640, 480)) 
            cv2.imshow("YOLOv8 Tracking"+str(id), annotated_frame)

           
            if len(to_be_reid) == len(reid_map[id].q) and len(to_be_reid) > 0:
                reid_target_images = []
                reid_init_images = []
                for key in to_be_reid:
                    reid_target_images.append((key,to_be_reid[key][0], to_be_reid[key][1])) #to be reid[key] has photo

                for key in reid_map[id].q: ##here camera source is not considered for now 
                    shared_person_key = reid_map[id].q[key].g_id
                    with shared_person_lock:
                        reid_init_images.append((shared_person_key,shared_person[shared_person_key].photo))

                
                to_be_reid.clear()
                reid_map[id].q.clear()

                #Pairing function gets pairing
                pairing = get_pairing(reidModel, reid_init_images, reid_target_images)



                for i in range(len(pairing)):
                    print(f"CAMERA {id} :: Person with L{reid_target_images[i][0]} has been reidentified as G{reid_init_images[i][0]}  ")
                    event_logger.write_log(log_file_path, f"Person {reid_init_images[i][0]} has exited camera {shared_person[reid_init_images[i][0]].camera_id} and entered camera {id}")
                    sqlite.record_entry(id,reid_init_images[i][0], conn)
                    update_person_location(reid_init_images[i][0], id )   
                    update_person_image(reid_init_images[i][0], reid_target_images[pairing[i]][1]) #update image to re ided one
                    
                    local_id_mapping[reid_target_images[pairing[i]][0]] = localMappingPerson(reid_init_images[i][0], STATUS_IDD, reid_target_images[pairing[i]][2])
                    
                         

                #print(id,key,to_be_reid[key])


                        #print("                                        ",str(id),bboxId(),directions[dir],  bbox.conf.item())
                    #q.put_nowait({"source": id, "destination": directions[dir], "person": bboxId(), "conf":bbox.conf.item()})
                
                    
                    #print({"source": id, "destination": get_dest, "person": bboxId(), "conf":bbox.conf.item()})    

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
    conn.close()

if __name__ == "__main__":
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default="RE_ID_VAE\saved\models\VAE_REID\important-models\checkpoint-epoch50.pth", type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    
    config = ConfigParser.from_args(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    reidModel = getModelFromConfig(config)

    record_output = False
    output_folder = "captured_frames"
    duration = None  # None - continuous recording || n - records for n seconds

    #Get graph
    graph_file_path = "mapper/network_map.txt"
    graph = create_graph_from_file(graph_file_path)
  
    n_camera = graph.number_of_nodes()
    # Communicate using the Queue
    reid_map = [ LockArray() for i in range(n_camera)]
    #n_camera = 1
    shared_person = {}  # Shared dictionary for person locations
    shared_person_lock = Lock()

    # Event Logger
    log_folder_path, log_file_path = event_logger.create_unique_folder_and_log()


    # Define the video files for the trackers
    args= [ (INPUT_FILE,"./test_data/1.mp4"),(INPUT_FILE,"./test_data/2.mp4"),(INPUT_FILE,"./test_data/3.mp4")]
    #args= [ (INPUT_WEBCAM,0),(INPUT_MOB,"http://192.168.50.151:8085/shot.jpg"),(INPUT_MOB,"http://192.168.50.67:8080/shot.jpg")]
    #args= [ (INPUT_WEBCAM,0),(INPUT_MOB,"http://192.168.50.151:8085/shot.jpg")]
    # Create the tracker threads
    threads = []
    for i in range(n_camera):
        threads.append(threading.Thread(target=analyze_footage, args=(args[i][0],args[i][1],record_output,output_folder, duration,i, reidModel), daemon=True))

    # Start the tracker threads
        
    for i in range(n_camera):
        threads[i].start()

    # Wait for the tracker threads to finish
    for i in range(n_camera):
        threads[i].join()

    # Clean up and close windows
    cv2.destroyAllWindows()