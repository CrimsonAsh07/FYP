from torch.utils.data import Dataset
import os
import torch
from torchvision import transforms
import PIL
import matplotlib.pyplot as plt
import argparse
import torch
from tqdm import tqdm
import model.model as module_arch
from parse_config import ConfigParser
from torch.nn import functional as F
from torchvision import transforms
from torch.autograd import Variable
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import time
class PairDataset(Dataset):
  def __init__(self, image_dir, device):
    self.image_dir = image_dir
    self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir)]
    self.transform = transforms.Compose([
        transforms.ToTensor(),  # Convert PIL image to tensor
    ])
    self.device = device

  def __len__(self):
    return len(self.image_paths) // 2  # Assuming each pair has 2 images

  def __getitem__(self, idx):
    # Get image paths for a random pair
    pair_idx = idx * 2  # Get index for the first image of the pair
    image1_path = self.image_paths[pair_idx]
    image2_path = self.image_paths[pair_idx + 1]
    #print(self.image_paths[pair_idx],self.image_paths[pair_idx + 1] )
    # Load images using PIL
    image1 = self.load_image(image1_path)
    image2 = self.load_image(image2_path)

    # Apply transformations (convert to tensor and normalize)
    image1 = self.transform(image1)
    image2 = self.transform(image2)

    image1 =image1.unsqueeze(0)
    image2 =image2.unsqueeze(0)

    image1 = image1.to(self.device)
    image2 = image2.to(self.device)
    # Extract type information from filename
    filename1 = os.path.basename(image1_path)
    pN = int(filename1.split("_")[0])

    return image1, image2, pN

  def load_image(self, image_path):
    """Loads an image using PIL."""
    image = PIL.Image.open(image_path).convert('RGB')  # Ensure RGB mode
    return image
  

def show_image(image_tensor):
    output_tensor = image_tensor.squeeze(0).cpu().detach().permute(1, 2, 0)
     # Permute to (height, width, channels)

    plt.imshow(output_tensor)  # Remove batch dimension, convert to CPU, detach
    plt.axis('off')  # Hide axes
    plt.show()


class Distance_Metric:
    
    def cosineMU(z1, mu1_1, logvar1, z2, mu1_2, logvar2, threshold, device,w_z=1/3, w_mu1=1/3, w_logvar=1/3):
        mu1 = mu1_1.to(device)
        mu2 = mu1_2.to(device)
        similarity = torch.nn.functional.cosine_similarity(mu1, mu2)
        return similarity > threshold, similarity
    
    def cosineZ(z1, mu1_1, logvar1, z2, mu1_2, logvar2, threshold, device,w_z=1/3, w_mu1=1/3, w_logvar=1/3):
        mu1 = z1.to(device)
        mu2 = z2.to(device)
        similarity = torch.nn.functional.cosine_similarity(mu1, mu2)
        return similarity > threshold, similarity
    
    def L2Z(z1, mu1_1, logvar1, z2, mu1_2, logvar2, threshold, device,w_z=1/3, w_mu1=1/3, w_logvar=1/3):

        tensor1 = z1.to(device)
        tensor2 = z2.to(device)

        # Calculate squared difference efficiently using broadcasting
        squared_diff = torch.pow(tensor1 - tensor2, 2)

        # Reduce across all dimensions (efficiently calculates sum of squared differences)
        euclidean_distance = torch.sqrt(torch.sum(squared_diff, dim=tuple(range(len(squared_diff.size())))))
        
        return euclidean_distance < threshold,euclidean_distance
    
    def L2M(z1, mu1_1, logvar1, z2, mu1_2, logvar2, threshold, device,w_z=1/3, w_mu1=1/3, w_logvar=1/3):

        tensor1 = mu1_1.to(device)
        tensor2 = mu1_2.to(device)

        # Calculate squared difference efficiently using broadcasting
        squared_diff = torch.pow(tensor1 - tensor2, 2)

        # Reduce across all dimensions (efficiently calculates sum of squared differences)
        euclidean_distance = torch.sqrt(torch.sum(squared_diff, dim=tuple(range(len(squared_diff.size())))))
        
        return euclidean_distance < threshold,euclidean_distance
    
    def weighted_similarity(z1, mu1_1, logvar1, z2, mu1_2, logvar2, threshold, device,w_z=1/3, w_mu1=1/3, w_logvar=1/3 ):
      """
      Calculates a weighted sum similarity metric between two images in the VAE latent space.

      Args:
          z1 (torch.Tensor): z vector of the first image.
          mu1_1 (torch.Tensor): mu1 vector of the first image.
          logvar1 (torch.Tensor): logvar vector of the first image.
          z2 (torch.Tensor): z vector of the second image.
          mu1_2 (torch.Tensor): mu1 vector of the second image.
          logvar2 (torch.Tensor): logvar vector of the second image.
          w_z (float): Weight for z distance (default: 1/3).
          w_mu1 (float): Weight for mu1 difference (default: 1/3).
          w_logvar (float): Weight for logvar difference (default: 1/3).

      Returns:
          float: Similarity score (lower score indicates higher similarity).
      """

      # Ensure all tensors have the same device
      z1 = z1.to(device)
      mu1_1 = mu1_1.to(device)
      logvar1 = logvar1.to(device)
      z2 = z2.to(device)
      mu1_2 = mu1_2.to(device)
      logvar2 = logvar2.to(device)

      # Calculate distance metrics
      z_dist = torch.linalg.norm(z1 - z2)
      mu1_diff = torch.abs(mu1_1 - mu1_2).sum()
      logvar_diff = torch.abs(logvar1 - logvar2).sum()

      # Weighted sum
      similarity = w_z * z_dist + w_mu1 * mu1_diff + w_logvar * logvar_diff

      return similarity< threshold, similarity
    

def main(config):
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

    total_loss = 0.0
    # total_metrics = torch.zeros(len(metric_fns))

    

    #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization

    #print(f"Shape of latent representation: {latent_representation_firstImage.shape}")
    distance_metric = Distance_Metric()
    distance_metric_function = getattr(Distance_Metric, config['distance_metric']['type'])
    distance_threshold = config['distance_metric']['Threshold']
    similarityType = config['distance_metric']['similarityType']
    print(config['distance_metric']['type'],config['distance_metric']['Threshold'] )


    data_file_names = os.listdir(config['data_loader']['args']['data_dir'])

    dataset = PairDataset(config['data_loader']['args']['data_dir'], device)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
    start = time.time()
    

    with torch.no_grad():
       total_samples = 0
       correct_predict = 0
       pnList = []
       for image1,image2, pN in dataloader:
          if(pN[0] == pN[1]):
             continue
          image1.to(device)
          image2.to(device)
          pnList.append(pN)
          latent_space1 = []
          latent_space2 = []
          for i in range(len(image1)):
             mu1, logvar1, z1 = model.encode(image1[i])
             mu2, logvar2, z2 = model.encode(image2[i])
             #show_image(image1[i])
             #show_image(image2[i])
             latent_space1.append([z1,mu1, logvar1])
             latent_space2.append([z2,mu2, logvar2])
          
      
            
          num_elements = len(latent_space1)
          operation_matrix = torch.zeros((num_elements, num_elements))

          minVal = float('inf')
          for i in range(num_elements):
            for j in range(num_elements):
              tensor1 = latent_space1[i]
              tensor2 = latent_space2[j]
              similarity ,operation_matrix[i, j] = distance_metric_function(tensor1[0], tensor1[1], tensor1[2],tensor2[0], tensor2[1], tensor2[2], distance_threshold, device)
              if(operation_matrix[i,j] < minVal):
                minVal = operation_matrix[i,j]

          #print(operation_matrix)
          #print(pN) 
          if(len(latent_space1) > 1 and len(latent_space2) > 1):      
            total_samples+=1
            if (similarityType == 1 and operation_matrix[0,0] + operation_matrix[1,1] > operation_matrix[0,1] + operation_matrix[1,0] ):
              correct_predict +=1
            elif (similarityType == 0 and operation_matrix[0,0] + operation_matrix[1,1] < operation_matrix[0,1] + operation_matrix[1,0] ):  
              correct_predict +=1
          else:
             print(pN)    

            
          
          
    end = time.time()
    runtime = end - start
    print("FPS: ", len(pnList)/runtime)
    print("Total Pairs tested: ",len(pnList), " Total Images Tested: ", len(pnList)*2)
    print("Correct Predictions: ", correct_predict)
    print("Accuracy: ", correct_predict/total_samples )
    print("Percentage Accuracy: ", round(correct_predict/total_samples*100,2),"%")

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
