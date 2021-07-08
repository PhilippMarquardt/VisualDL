from torch.utils.data import DataLoader, Dataset
import yaml
import logging
from skimage import io
import cv2
from itertools import chain, combinations
import albumentations as A
import numpy as np
import torch
from timm import list_models

def write_image(path, src):
    cv2.imwrite(path, src*255)

    
def get_dataloader(dataset:Dataset, batch_size:int, workers:int, shuffle:bool = True) -> DataLoader:
    """
    Return a dataloader for a dataset with specific settings

    Args:
        dataset (Dataset): The dataset for the DataLoader
        batch_size (int): The batch size
        workers (int): Number of workers
        shuffle (bool, optional): Whether it will be shuffled. Defaults to True.

    Returns:
        DataLoader: [description]
    """
    return DataLoader(dataset, batch_size = batch_size, shuffle = shuffle, num_workers = workers)
    

def parse_yaml(yaml_file:str) -> dict:
    """Parses a yaml file

    Args:
        yaml_file (str): Path to the yaml file

    Returns:
        dict: The parsed yaml file
    """
    if type(yaml_file) is dict:
        return yaml_file
    with open(yaml_file, "r") as handle:
        return yaml.load(handle, Loader=yaml.FullLoader)

def timm_universal_encoders(**kwargs):
    timm_universal_prefix = 'timm-u-'
    return [f'{timm_universal_prefix}{i}' for i in list_models(**kwargs)]

def get_all_combinations(li:list):
    """Returns a list(tuple) containing all combinations of the given list.

    Args:
        li (list): The input list.
    """
    return list(chain(*map(lambda x: combinations(li, x), range(1, len(li)+1))))

def get_weight_map(imgs):
    """Gets a weight map for mask

    Args:
        img (np.array): mask np array (B,C,W,H)
    """
    weight_maps = []
    for img in imgs:
        img = cv2.cvtColor(img.astype('float32'), cv2.COLOR_GRAY2BGR)
        kernel = np.ones((3, 3), 'uint8')
        dilate_img = cv2.dilate(img, kernel, iterations=1)
        img1_bg = dilate_img - img
        img1 = img1_bg[:,:,0]
        clipper = np.clip(img1, 1, 8)
        weight_maps.append(clipper) # weight edges by factor (e.g. 6)
    weight_maps = torch.tensor(weight_maps)
    return weight_maps
def parse_classification_config(config_path):
    pass

def get_transform_from_config(cfg:dict):
    """Parses the config into Albumentation transforms.

    Args:
        cfg (dict): The config.
    """
    transforms = []
    valid_trans = []
    for key, val in cfg['transforms'].items():
        if key == "Resize":
            valid_trans.append(eval(f"A.{key}({val})"))
        transforms.append(eval(f"A.{key}({val})"))
    if len(valid_trans) == 0:
        valid_trans = None
    return A.Compose(transforms), A.Compose(valid_trans)
                 
        



    
    
    
    
