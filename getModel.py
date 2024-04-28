import argparse
import torch
from model import model as module_arch
from parse_config import ConfigParser
import torchvision.transforms as transforms
#Fixes PosixPath Error
import pathlib
import cv2
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
from scipy.optimize import linear_sum_assignment

transformer = transforms.ToTensor()
def L2M(z1, mu1_1, logvar1, z2, mu1_2, logvar2, threshold, device,w_z=1/3, w_mu1=1/3, w_logvar=1/3):

        tensor1 = mu1_1.to(device)
        tensor2 = mu1_2.to(device)

        # Calculate squared difference efficiently using broadcasting
        squared_diff = torch.pow(tensor1 - tensor2, 2)

        # Reduce across all dimensions (efficiently calculates sum of squared differences)
        euclidean_distance = torch.sqrt(torch.sum(squared_diff, dim=tuple(range(len(squared_diff.size())))))
        
        return euclidean_distance < threshold,euclidean_distance

def getModelFromConfig(config):

    logger = config.get_logger('test')

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)
    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    # checkpoint = torch.load(config.resume)

    # loading on CPU-only machine
    checkpoint = torch.load(config.resume, map_location=torch.device('cpu'))
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    return model

def get_tensor(device, crop_object):
    image = cv2.cvtColor(crop_object, cv2.COLOR_BGR2RGB)
    resized_image = cv2.resize(image, (96, 224), interpolation=cv2.INTER_AREA)
    tensor = transformer(resized_image)
    tensor = tensor.unsqueeze(0)
    tensor = tensor.to(device)
    return tensor

def get_pairing(model, init_images, target_images):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    init_tensors = []
    target_tensors = []
    
    for i in range(len(init_images)):
        init_tensors.append(get_tensor(device, init_images[i][1]))
        target_tensors.append(get_tensor(device, target_images[i][1]))

    latent_space1 = [] 
    latent_space2= []

    for i in range(len(init_tensors)):
        mu1, logvar1, z1 = model.encode(init_tensors[i])
        mu2, logvar2, z2 = model.encode(target_tensors[i])
             #show_image(image1[i])
             #show_image(image2[i])
        latent_space1.append([z1,mu1, logvar1])
        latent_space2.append([z2,mu2, logvar2])


    num_elements = len(latent_space1)
    operation_matrix = torch.zeros((num_elements, num_elements))
    
    for i in range(num_elements):
        for j in range(num_elements):
            tensor1 = latent_space1[i]
            tensor2 = latent_space2[j]
            
            similarity ,distance = L2M(tensor1[0], tensor1[1], tensor1[2],tensor2[0], tensor2[1], tensor2[2], 25, device)
            operation_matrix[i, j] = distance.item()
    pairing = linear_sum_assignment(operation_matrix)
    row_pair,col_pair = linear_sum_assignment(operation_matrix)              

    return col_pair
    

    

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    getModelFromConfig(config)
