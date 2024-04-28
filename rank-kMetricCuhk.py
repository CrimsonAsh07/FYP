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
import pickle
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
        #print("Preprocessed Image Shape", preprocessed_image.shape)
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
    

    #print(f"Shape of latent representation: {latent_representation_firstImage.shape}")
    distance_metric = Distance_Metric()
    distance_metric_function = getattr(Distance_Metric, config['distance_metric']['type'])
    distance_threshold = config['distance_metric']['Threshold']
    print(config['distance_metric']['type'],config['distance_metric']['Threshold'] )
    
    data_dir = "RE_ID_VAE/data-cuhk/images_labeled"
    data_file_names = os.listdir(data_dir)

    def count_Percent_matching_ids(distances, k, id_initial):
        """
        Counts the number of elements with id=id_initial among the first k elements in a sorted list of tuples.

        Args:
            distances: A list of sorted tuples (distance, id) pairs.
            k: The number of elements to consider from the beginning of the sorted list.
            id_initial: The target id to search for.

        Returns:
            An integer representing the count of matching ids within the first k elements.
        """
        count = 0
        countTotal = 0
        for i in range(min(k, len(distances))):  
            countTotal+=1# Limit loop to k or list length
            if distances[i][1] == id_initial:
                count += 1
             # Stop counting if id mismatch is found

        if countTotal == 0:
            return 0
        return (count/countTotal)*100
    
    def count_rank_k(distances, k, id_initial ):
        """
        Counts the number of elements with id=id_initial among the first k elements in a sorted list of tuples.

        Args:
            distances: A list of sorted tuples (distance, id) pairs.
            k: The number of elements to consider from the beginning of the sorted list.
            id_initial: The target id to search for.

        Returns:
            An integer representing the count of matching ids within the first k elements.
        """
        count = 0
        countTotal=0
        length =  len(distances)
        for i in range(min(k,length)):  
            countTotal+=1# Limit loop to k or list length
            if distances[i][1] == id_initial:
                count = 1
                break
             # Stop counting if id mismatch is found

        if countTotal == 0:
            return 0
        return count


    def count_Percent_matching_ids(distances, k, id_initial):
        """
        Counts the number of elements with id=id_initial among the first k elements in a sorted list of tuples.

        Args:
            distances: A list of sorted tuples (distance, id) pairs.
            k: The number of elements to consider from the beginning of the sorted list.
            id_initial: The target id to search for.

        Returns:
            An integer representing the count of matching ids within the first k elements.
        """
        count = 0
        countTotal = 0
        for i in range(min(k, len(distances))):  
            countTotal+=1# Limit loop to k or list length
            if distances[i][1] == id_initial:
                count += 1
             # Stop counting if id mismatch is found

        if countTotal == 0:
            return 0
        return (count/countTotal)

    ranks = [5]
    rank_k_percent = []
   
    # Get the latent representation for the input image
    latent_space = []
    cc = 0
    num = 642
    with torch.no_grad():
        for j, (data, target) in enumerate(data_loader):
            if cc>=num:
                break
            cc+=1
            data, target = data.to(device), target.to(device)

            mu, logvar, z = model.encode(data)
            latent_space.append(z)

    # print("LS")
    # print(len(latent_space))
    # with open("tensors.pkl", "wb") as f:
    #     pickle.dump(latent_space, f)
    
    # with open('tensors.pkl', 'rb') as f:
    #     latent_space = pickle.load(f)
        
    for i in range(num):
        image_id = int(data_file_names[i].split("_")[1])
        #print(image_id, end=" ")
        distances = []
        idjs = []
        for j in range(num):
            if(i == j):
                continue
            image_idj = int(data_file_names[j].split("_")[1])
            similar, distance = distance_metric_function(latent_space[i],latent_space[j],distance_threshold, device='cuda' )
            distances.append((distance.cpu().item(),image_idj))
            
        
    
                    
        sorted_dist = sorted(distances, key=lambda x: x[0])
        
        k=10
  # Find indices of k largest elements in the first values array
        #print(distances)
        
        
  
        # Select the corresponding tuples from the original data using the indices
        
        rankValTuple = []
        for rank in ranks:
            rankVal = count_Percent_matching_ids(sorted_dist, rank, image_id)
            rankValTuple.append(rankVal)
        rank_k_percent.append(rankValTuple)
        print("d",end=" ")
    print(rank_k_percent)
    maxKs = np.max(rank_k_percent,axis=0)
    meanKs = np.mean(rank_k_percent,axis=0)
    
    #print(f"Rank-1 : {meanKs[0]*100 :.4f}")
    #print(f"Rank-5 : {meanKs[1]*100:.4f}")
    #print(f"Rank-10 : {meanKs[2]*100:.4f}")
    print(f"Precision: {maxKs[0]*100:.4f}")
    print(f"mPA: {meanKs[0]*100:.4f}")

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
