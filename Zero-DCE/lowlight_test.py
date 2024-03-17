import shutil
import cv2
import torch
import os
import sys
import time
from PIL import Image
import glob
import numpy as np
import model
from torchvision import transforms
import torchvision

def is_dark(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    average_brightness = cv2.mean(gray_image)[0]

    threshold = 100

    print("Average Intensity", average_brightness)  

    return average_brightness < threshold

def lowlight(image_path, threshold=100):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    data_lowlight = Image.open(image_path)
    
    if is_dark(image_path):

        data_lowlight = (np.asarray(data_lowlight) / 255.0)
        data_lowlight = torch.from_numpy(data_lowlight).float()
        data_lowlight = data_lowlight.permute(2, 0, 1)
        data_lowlight = data_lowlight.cuda().unsqueeze(0)

        DCE_net = model.enhance_net_nopool().cuda()
        DCE_net.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__),'snapshots', 'Epoch99.pth')))

        start = time.time()
        _, enhanced_image, _ = DCE_net(data_lowlight)

        end_time = (time.time() - start)
        print("Processing Time:", end_time)

        result_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'enhanced_frames'))
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        result_path = os.path.join(result_dir, os.path.basename(image_path))

        torchvision.utils.save_image(enhanced_image, result_path)

        print("Enhanced", os.path.basename(image_path)) 

    else:
        
        result_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'enhanced_frames'))
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        result_path = os.path.join(result_dir, os.path.basename(image_path))

        shutil.copyfile(image_path, result_path)

        print("Skipped", os.path.basename(image_path))


def enhance_images(input_folder):
    processed_images = set()

    image_files_path = glob.glob(os.path.join(input_folder, '*.jpg'))
    processed_images.update(image_files_path)

    while True:
        image_files_path = glob.glob(os.path.join(input_folder, '*.jpg'))

        for image_path in image_files_path:
            if image_path not in processed_images:
                lowlight(image_path)
                print("-----------") 
                processed_images.add(image_path)

        time.sleep(1)
        
if __name__ == '__main__':

    input_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'captured_frames'))
    
    enhance_images(input_path)


