from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch
import cv2
import numpy as np
from tqdm import tqdm
from .utils import get_all_combinations

def visualize(model, layer, image):
        cam = GradCAM(model=model, target_layer=layer, use_cuda=torch.cuda.is_available())
        heatmap = cv2.applyColorMap(np.uint8(255 * cam(input_tensor=image.unsqueeze(0))[0,:]), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + image.permute(1,2,0).numpy()
        cam = cam / np.max(cam)
        return cam


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
        if valid_loader:
            valid_bar = tqdm(valid_loader)
            evaluate(model, valid_bar, criterions=criterions, criterion_scaling=criterion_scaling, writer=writer, metric=monitor_metric, device=device,
            epoch=epoch, name = name, average_outputs=False)

         
        
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
                
                
    
def evaluate(model, valid_bar, criterions, criterion_scaling, writer, metric, device, epoch, name, average_outputs = False):
    assert writer is not None
    assert metric is not None
    metric.reset()

    total_loss = 0.0    
    for cnt, (x,y) in enumerate(valid_bar):
        x = x.to(device)
        y = y.to(device)
        model.zero_grad()
        #TODO implement average_outputs
        with torch.cuda.amp.autocast():
            loss = None
            with torch.no_grad():
                predictions = model(x)
            for cr, scal in zip(criterions, criterion_scaling):
                if loss is None:
                    loss = cr(predictions, y) / scal
                else:
                    loss += cr(predictions, y) / scal
            predictions = torch.argmax(predictions, 1)
        metric.update(predictions.detach().cpu(), y.detach().cpu())
        
        metric_str = f"Valid: Epoch:%i, Loss:%.4f, {metric.__class__.__name__}:%.4f"
        metric_value = metric.compute()
        writer.add_scalar(f"valid/valid-{name}-{metric.__class__.__name__}", metric_value, epoch)
        
        total_loss += loss.item()
        current_loss = total_loss / float((cnt+1))
        valid_bar.set_description(metric_str % tuple([epoch, current_loss, metric_value]))   
    metric.reset() 

def test_trainer(models: list, test_loaders, metric):
    assert test_loaders
    assert metric
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    metric.reset()
    log_dict = {}
    combinations = get_all_combinations(models)
    for cnt, model_comb in enumerate(tqdm(combinations)):
        names = ",".join([x.name for x in model_comb])
        predictions = None     
        for (model, test_loader) in zip(model_comb, test_loaders):
            for (x,y) in test_loader:
                x = x.to(device)
                y = y.to(device)
                with torch.cuda.amp.autocast():
                    with torch.no_grad():
                        if predictions is None:
                            predictions = model(x)
                        else:
                            predictions += model(x)
        prediction = torch.argmax(predictions, 1)
        metric.update(prediction.detach().cpu(), y.detach().cpu())
        val = metric.compute()
        log_dict[names] = val
    return log_dict
                
