from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
import torch
import yaml
import logging
from torchmetrics import * 
from skimage import io
import cv2

def train_all_epochs(model, train_loader, valid_loader, test_loader, epochs, criterions, metrics, writer, criterion_scaling = None, average_outputs = False, name:str = "", monitor_metric = None):
    
    if criterion_scaling is None:
        criterion_scaling = [1] * len(criterions)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    scaler = torch.cuda.amp.GradScaler(enabled = False if device == 'cpu' else True)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.0001)
    model = model.to(device)
    
    for epoch in range(epochs):
        training_bar = tqdm(train_loader)
        train_one_epoch(model, training_bar, criterions, criterion_scaling, average_outputs, device, epoch, optimizer, scaler, metrics, writer, name)
        evaluate(model, valid_loader, criterions=criterions, criterion_scaling=criterion_scaling, writer=writer, metric=monitor_metric ,average_outputs=False)
         
        



def train_one_epoch(model, training_bar, criterions, criterion_scaling, average_outputs = False, device = None, epoch = 0, optimizer = None, scaler = None, metrics = None, writer = None, name:str = ""):
    for metric in metrics:
        metric.reset() 
    total_loss = 0.0    
    for cnt, (x,y) in enumerate(training_bar):
        x = x.to(device)
        y = y.to(device)
        model.zero_grad()
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
            metric.update(predictions.detach().cpu(), y.detach().cpu())
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        metric_str = "Train: Epoch:%i, Loss:%.4f, " + "".join([metric.__class__.__name__ + ":%.4f, " for metric in metrics ])
        epoch_values = [metric.compute() for metric in metrics]
        for metric, val in zip(metrics, epoch_values):
            writer.add_scalar(f"train/train-{name}-{metric.__class__.__name__}", val, epoch)
        
        total_loss += loss.item()
        current_loss = total_loss / float((cnt+1))
        training_bar.set_description(metric_str % tuple([epoch, current_loss]+epoch_values))   

    
    for metric in metrics:
        metric.reset() 
                
                
    
def evaluate(model, loader, criterions, criterion_scaling, writer, metric, average_outputs = False, ):
    assert writer is not None
    assert metric is not None
    metric.reset()

    
    pass


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






    
    
    
    
