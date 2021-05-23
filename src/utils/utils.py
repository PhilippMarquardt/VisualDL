from torch.utils.data import DataLoader, Dataset
import yaml
import logging
from skimage import io
import cv2
from itertools import chain, combinations





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
    with open(yaml_file, "r") as handle:
        return yaml.load(handle, Loader=yaml.FullLoader)

def get_all_combinations(li:list):
    """Returns a list(tuple) containing all combinations of the given list.

    Args:
        li (list): The input list.
    """
    return list(chain(*map(lambda x: combinations(li, x), range(1, len(li)+1))))





    
    
    
    
