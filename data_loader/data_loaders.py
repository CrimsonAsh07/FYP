from torchvision import datasets, transforms
from base import BaseDataLoader


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(
            self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class CelebDataLoader(BaseDataLoader):
    """
    CelebA data loading
    Download and extract:
    https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/celeba.zip 
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, image_size=64):
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ])
        self.data_dir = data_dir
        self.dataset = datasets.ImageFolder(
            self.data_dir, transform=transform)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class ReidDataLoader(BaseDataLoader):
    """
    ReID DUKEMTMC data loading
    Download and extract to data folder:
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1):
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.data_dir = data_dir
        self.dataset = datasets.ImageFolder(
            self.data_dir, transform=transform)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class ReidDataLoaderCUHK(BaseDataLoader):
    """
    ReID CUHK data loading
    Download and extract to data folder:
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.data_dir = data_dir
        self.dataset = datasets.ImageFolder(
            self.data_dir, transform=transform)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)        
