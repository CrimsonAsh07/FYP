
import cv2
import torch
import os
import time
from PIL import Image
import glob
import numpy as np
from Zero_DCE import model
from torchvision import transforms


def is_dark(image, threshold, verbose):

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    average_brightness = cv2.mean(gray_image)[0]
    if verbose: 
        print("Average Intensity", average_brightness)  

    return average_brightness < threshold

def lowlight(frame, threshold=100, verbose = True):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    data_lowlight = Image.fromarray(rgb_frame)
    
    if is_dark(frame, threshold, verbose):

        data_lowlight = (np.asarray(data_lowlight) / 255.0)
        data_lowlight = torch.from_numpy(data_lowlight).float()
        data_lowlight = data_lowlight.permute(2, 0, 1)
        data_lowlight = data_lowlight.cuda().unsqueeze(0)

        DCE_net = model.enhance_net_nopool().cuda()
        DCE_net.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__),'snapshots', 'Epoch199.pth')))

        start = time.time()
        _, enhanced_image, _ = DCE_net(data_lowlight)

        end_time = (time.time() - start)
        if verbose:  
            print("Processing Time:", end_time)

        # result_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'enhanced_frames'))
        # if not os.path.exists(result_dir):
        #     os.makedirs(result_dir)
        # result_path = os.path.join(result_dir, os.path.basename(image_path))

        # torchvision.utils.save_image(enhanced_image, result_path)

        # print("Enhanced", os.path.basename(image_path)) 
        
        
        numpy_image = enhanced_image.detach().cpu().numpy()
        numpy_image = np.asarray(numpy_image).reshape(numpy_image.shape[1:])
                    # print(numpy_image.shape)
        numpy_image = np.transpose(numpy_image, (1, 2, 0))
        cv2_image = cv2.cvtColor(numpy_image, cv2.COLOR_BGR2RGB)  
        cv2_image = cv2_image * 255
        cv2_image = cv2_image.astype(np.uint8)
        
        return cv2_image
    else:
        if verbose: 
            print ("Skipped as it is bright!")
        return frame



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
        
# if __name__ == '__main__':

#     input_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'captured_frames'))
    
#     enhance_images(input_path)


