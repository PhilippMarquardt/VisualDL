from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch
import cv2
import numpy as np
from tqdm import tqdm
from .utils import get_all_combinations, get_weight_map
from torchmetrics import ConfusionMatrix
import logging
import os
import sys


def visualize(model, layer, image):
        cam = GradCAM(model=model, target_layer=layer, use_cuda=torch.cuda.is_available())
        heatmap = cv2.applyColorMap(np.uint8(255 * cam(input_tensor=image.unsqueeze(0))[0,:]), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + image.permute(1,2,0).numpy()
        cam = cam / np.max(cam)
        return cam


def train_all_epochs(model, train_loader, valid_loader, test_loader, epochs, criterions, metrics, monitor_metric, writer, optimizer, accumulate_batch, criterion_scaling = None, average_outputs = False, name:str = "", weight_map = False, save_folder = "", early_stopping = 10, modelstring = ""):
    #criterions = [torch.nn.CrossEntropyLoss(reduction="none")]
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    scaler = torch.cuda.amp.GradScaler(enabled = False if device == 'cpu' else True)
    model = model.to(device)
    accumulate_every = accumulate_batch // train_loader.batch_size
    best_metric = float("-inf")
    cnt = 0
    for epoch in range(epochs):
        training_bar = tqdm(train_loader, file=sys.stdout)
        train_one_epoch(model, training_bar, criterions, criterion_scaling, average_outputs, device, epoch, optimizer, scaler, metrics, writer, name, accumulate_every, best_metric, weight_map)
        if valid_loader:
            valid_bar = tqdm(valid_loader, file=sys.stdout)
            tmp = evaluate(model, valid_bar, criterions=criterions, criterion_scaling=criterion_scaling, writer=writer, metrics=metrics, monitor_metric=monitor_metric, device=device,
            epoch=epoch, name = name, average_outputs=False)
            if best_metric <= tmp:
                best_metric = tmp
                torch.save(model.state_dict(), os.path.join(save_folder, name + ".pt"))
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'model':modelstring,
                    'best_metric':best_metric}, os.path.join(save_folder, name + ".pt"))
                cnt = 0
            else:
                cnt +=1
            if cnt >= early_stopping:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}, os.path.join(save_folder, name + "last.pt"))
                model.load_state_dict(torch.load(os.path.join(save_folder, name + ".pt"))['model_state_dict'])
                return
                

         
        
def train_one_epoch(model, training_bar, criterions, criterion_scaling, average_outputs = False, device = None, epoch = 0, optimizer = None, scaler = None, metrics = None, writer = None, name:str = "", accumulate_every = 1, best_metric = 0.0, weight_map = False):
    for metric in metrics:
        metric.reset() 
    total_loss = 0.0    
    for cnt, (x,y) in enumerate(training_bar):
        x = x.to(device)
        y = y.to(device)
        #TODO implement average_outputs
        with torch.cuda.amp.autocast():
            loss = None
            try:
                predictions = model(x)
            except:
                continue
            #for cr, scal in zip(criterions, criterion_scaling):
                #cr = torch.nn.CrossEntropyLoss(reduction="none")
            #    if loss is None:
            #        loss = cr(predictions, y) 
            #    else:
            #        tmp = cr(predictions, y) 
            #        loss += tmp
            #TODO: add weight map here
            weight_maps = get_weight_map(y.detach().cpu().numpy() * 255.).to(device) if weight_map else None
            #for cnt, map in enumerate(weight_maps):
            #    map[map == 1] = 0
            #    cv2.imwrite(f"{cnt}.png", map.detach().cpu().numpy()  * 255.)
            loss = criterions(predictions, y, weight_maps)
            #weight_maps = get_weight_map(y.detach().cpu().numpy() * 255.).to(device)
            #loss *= weight_maps
            #for cnt, yy in enumerate(y):
            #    cv2.imwrite(f"{cnt}.png", yy.detach().cpu().numpy() * 255)



            #loss = loss.mean()
            predictions = torch.argmax(predictions, 1)
        for metric in metrics:
            metric.update(predictions.detach().cpu(), y.detach().cpu())
        scaler.scale(loss).backward()
        #gradient accumulation
        if (cnt > 0 and cnt % accumulate_every == 0) or cnt == len(training_bar) - 1:
            scaler.step(optimizer)
            scaler.update()
            model.zero_grad()

        metric_str = "Train: Epoch:%i, Loss:%.4f, Best:%.4f " + "".join([metric.__class__.__name__ + ":%.4f, " for metric in metrics ])
        epoch_values = [metric.compute().item() for metric in metrics]


        for metric, val in zip(metrics, epoch_values):
            writer.add_scalar(f"train/train-{name}-{metric.__class__.__name__}", val, epoch)
        
        
        total_loss += loss.item()
        current_loss = total_loss / float((cnt+1))
        writer.add_scalar(f"train/train-loss", current_loss, epoch)
        training_bar.set_description(metric_str % tuple([epoch+1, current_loss, best_metric]+epoch_values))   

    
    for metric in metrics:
        metric.reset() 
                
                
    
def evaluate(model, valid_bar, criterions, criterion_scaling, writer, metrics, monitor_metric, device, epoch, name, average_outputs = False):
    assert writer is not None
    assert metrics is not None
    assert monitor_metric is not None
    monitor_metric.reset()
    for metric in metrics:
        metric.reset()
    model.eval()
    total_loss = 0.0    
    for cnt, (x,y) in enumerate(valid_bar):
        x = x.to(device)
        y = y.to(device)
        model.zero_grad()
        #TODO implement average_outputs
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                try:
                    predictions = model(x)
                except:
                    continue
            loss = criterions(predictions, y)
            predictions = torch.argmax(predictions, 1)
        monitor_metric.update(predictions.detach().cpu(), y.detach().cpu())
        for metric in metrics:
            metric.update(predictions.detach().cpu(), y.detach().cpu())
        metric_str = "Valid: Epoch:%i, Loss:%.4f, " + "".join([metric.__class__.__name__ + ":%.4f, " for metric in metrics ])
        epoch_values = [metric.compute().item() for metric in metrics]

        for metric, val in zip(metrics, epoch_values):
            writer.add_scalar(f"valid/valid-{name}-{metric.__class__.__name__}", val, epoch)
        
        total_loss += loss.item()
        current_loss = total_loss / float((cnt+1))
        writer.add_scalar(f"valid/valid-loss", current_loss, epoch)
        valid_bar.set_description(metric_str % tuple([epoch+1, current_loss]+epoch_values))     
        
    for metric in metrics:
        metric.reset()
    model.train()

    #return total_loss / len(valid_bar)
    return monitor_metric.compute()

def test_trainer(models: list, test_loaders, metrics):
    assert test_loaders
    assert metrics
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    log_dict = {}
    combinations = get_all_combinations(models) #get all permutations of the model list
    for cnt, model_comb in enumerate(tqdm(combinations, file=sys.stdout)):
        for metric in metrics:
            metric.reset()
        names = ",".join([x.name for x in model_comb])    
        for (x,y) in test_loaders[0]:
            predictions = None 
            x = x.to(device)
            y = y.to(device)
            for model in model_comb:
                model.model.eval()
                with torch.cuda.amp.autocast():
                    with torch.no_grad():
                        if predictions is None:
                            predictions = model(x).detach().cpu()
                        else:
                            predictions += model(x).detach().cpu()
            predictions = torch.argmax(predictions, 1)
            for metric in metrics:
                metric.update(predictions.detach().cpu(), y.detach().cpu())
        log_dict[names] = [metric.compute().item() for metric in metrics]
    return log_dict
                


def predict_images(model, images, device):
    model.eval()
    total_loss = 0.0 
    model = model.to(device)
    all_predictions = []
    for cnt, image in enumerate(images):
        image = image / 255.
        image = torch.unsqueeze(torch.tensor(image, dtype = torch.float).permute(2, 0, 1), 0)
        image = image.to(device)
        model.zero_grad()
        #TODO implement average_outputs
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                predictions = model(image)
            predictions = torch.argmax(predictions, 1)
            all_predictions.append(predictions[0].detach().cpu().numpy())

    return all_predictions
