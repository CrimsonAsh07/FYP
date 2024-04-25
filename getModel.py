import argparse
import torch
from model import model as module_arch
from parse_config import ConfigParser

#Fixes PosixPath Error
import pathlib

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath


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
