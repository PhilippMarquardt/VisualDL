from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
import torch
import yaml
import logging
from torchmetrics import * 

def train_all_epochs(model, train_loader, valid_loader, test_loader, epochs, criterions, metrics, criterion_scaling = None, average_outputs = False):
    
    if criterion_scaling is None:
        criterion_scaling = [1] * len(criterions)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    scaler = torch.cuda.amp.GradScaler(enabled = False if device == 'cpu' else True)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.0001)
    model = model.to(device)
    
    for epoch in range(epochs):
        training_bar = tqdm(train_loader)
        train_one_epoch(model, training_bar, criterions, criterion_scaling, average_outputs, device, epoch, optimizer, scaler, metrics)
        for metric in metrics:
            metric.reset() 
        



def train_one_epoch(model, training_bar, criterions, criterion_scaling, average_outputs = False, device = None, epoch = 0, optimizer = None, scaler = None, metrics = None): 
    for cnt, (x,y) in enumerate(training_bar):
        x = x.to(device)
        y = y.to(device)
        model.zero_grad()
        acc = 0.0
        #TODO implement average_outputs
        with torch.cuda.amp.autocast():
            loss = None
            predictions = model(x)
            for cr, scal in zip(criterions, criterion_scaling):
                if loss is None:
                    loss = cr(predictions, y) / scal
                else:
                    loss += cr(predictions, y) / scal
            predictions = torch.argmax(predictions, 1)
        for metric in metrics:
            acc = metric(predictions.detach().cpu(), y.detach().cpu())
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        metric_str = "Train: Epoch:%i, Loss:%.4f, " + "".join([metric.__class__.__name__ + ":%.4f, " for metric in metrics ])
        training_bar.set_description(metric_str % tuple([epoch, loss.item()]+[metric.compute() for metric in metrics]))   
           
                
                
    
def evaluate(loader, criterions, criterion_scaling, average_outputs = False):
    pass
    
    
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






    
    
    
    
