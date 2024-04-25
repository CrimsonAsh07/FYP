import argparse
import numpy as np
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from torch.nn import functional as F
import torchvision.utils as vutils
from torchvision import transforms
from torch.autograd import Variable
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from PIL import Image  # Import for image processing

#Fixes PosixPath Error
import pathlib

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

def show_image(image_tensor):
    output_tensor = image_tensor.squeeze(0).cpu().detach().permute(1, 2, 0)
     # Permute to (height, width, channels)

    plt.imshow(output_tensor)  # Remove batch dimension, convert to CPU, detach
    plt.axis('off')  # Hide axes
    plt.show()



def get_latent_representation(input_image_path, device, model):
        """
        Extracts the latent space representation for a given image path.

        Args:
            input_image_path (str): Path to the input image file (JPEG).

        Returns:
            torch.Tensor: The latent space representation (mu) tensor.
        """
        transform = transforms.Compose([  # Adjust based on model input size
            transforms.ToTensor(),
        ])
        # Load image using PIL
        img = Image.open(input_image_path).convert('RGB')  # Assuming RGB image

        # Preprocess the image (e.g., resize)
        preprocessed_image = transform(img)  # Replace with your transformation

        # Convert to PyTorch tensor
        #preprocessed_image = torch.from_numpy(np.asarray(preprocessed_image)).float()

        # Normalize the tensor (if needed)
        #preprocessed_image = normalize(preprocessed_image)  # Replace with your normalization

        # Add batch dimension (assuming model expects batches)
        preprocessed_image = preprocessed_image.unsqueeze(0)

        # Move the image to the appropriate device
        preprocessed_image = preprocessed_image.to(device)

        # Encode the image to get the latent representation
        print("Preprocessed Image Shape", preprocessed_image.shape)
        with torch.no_grad():
            mu, logvar, z = model.encode(preprocessed_image)

        # Extract the latent representation (mu in this case)
        
        # recons = model.decode(z)
        # vutils.save_image(recons.data,
        #                   os.path.join(
        #                       "Reconstructions",
        #                       f"recons_Image1_epoch_{config['trainer']['epochs']}.png"),
        #                   normalize=True,
        #                   nrow=1)
        return mu, logvar, z

    # Define transformations (replace with your specific preprocessing steps)
    
class Distance_Metric:
    
    def cosine(mu1, mu2, threshold, device):
        mu1.to(device), mu2.to(device)
        similarity = torch.nn.functional.cosine_similarity(mu1, mu2)
        return similarity > threshold, similarity

    def L2(tensor1, tensor2, threshold, device):

        tensor1 = tensor1.to(device)
        tensor2 = tensor2.to(device)

        # Calculate squared difference efficiently using broadcasting
        squared_diff = torch.pow(tensor1 - tensor2, 2)

        # Reduce across all dimensions (efficiently calculates sum of squared differences)
        euclidean_distance = torch.sqrt(torch.sum(squared_diff, dim=tuple(range(len(squared_diff.size())))))

        return euclidean_distance < threshold,euclidean_distance
    
def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=1,
        shuffle=False,
        validation_split=0.0,
        # training=False,
        num_workers=2
    )

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    # metric_fns = [getattr(module_metric, met) for met in config['metrics']]

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

    total_loss = 0.0
    # total_metrics = torch.zeros(len(metric_fns))

    

    #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization

    # Example usage:
    input_image_path = "data/bounding_box_test/0002_c1_f0044158.jpg"
    data_dir = "data/bounding_box_test/"
    # Get the latent representation for the input image
    mu0, logvar0, z0 = get_latent_representation(input_image_path, device, model)

    #print(f"Shape of latent representation: {latent_representation_firstImage.shape}")
    distance_metric = Distance_Metric()
    distance_metric_function = getattr(Distance_Metric, config['distance_metric']['type'])
    distance_threshold = config['distance_metric']['Threshold']
    print(config['distance_metric']['type'],config['distance_metric']['Threshold'] )
    data_file_names = os.listdir(data_dir)

    countT = 0
    countS = 0
    diss = 0
    print(data_loader.shape)
    with torch.no_grad():
        for i, (data, target) in enumerate(data_loader):
            #show_image(data)
            data, target = data.to(device), target.to(device)
            mu, logvar, z = model.encode(data)
            similar, distance = distance_metric_function(z, z0,distance_threshold, device='cuda' )
            

                
            if similar:
                countS+=1
                print("distance",  distance.item(), i, data_file_names[i])
            countT+=1
            
            

    print(countS, countT)

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    

    config = ConfigParser.from_args(args)
    main(config)
