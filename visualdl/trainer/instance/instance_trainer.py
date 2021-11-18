from ...utils.datasets import InstanceSegmentationDataset
from ...utils.utils import *
from ...utils.utils import parse_yaml, get_transform_from_config, get_dataloader
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm
import os
import sysconfig
import sys
import subprocess
import argparse


class InstanceTrainer():
    def __init__(self, cfg_path:dict):
        self.cfg = parse_yaml(cfg_path)
        train_trans, val_trans = get_transform_from_config(cfg=self.cfg)
        trainset = InstanceSegmentationDataset(folder=self.cfg['data']['train'], transform = train_trans)
        validset = InstanceSegmentationDataset(folder=self.cfg['data']['valid'], transform = val_trans)
        savefolder = self.cfg['data']['save_folder']
        weight_path = self.cfg['data']['weights']
        # load a model pre-trained on COCO
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True, box_detections_per_img = self.cfg['settings']['max_boxes_per_image'], min_size = 512, trainable_backbone_layers=5)
        modelstring = f"torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained={True}, box_detections_per_img = {self.cfg['settings']['max_boxes_per_image']})"
        if os.path.isfile(weight_path):
            try:
                model.load_state_dict(torch.load(self.cfg['data']['weights'])['model_state_dict'], strict=False)
                print("Weights loaded!")
            except:
                print("Could not load weights!")
        loader = get_dataloader(trainset, int(self.cfg['settings']['batch_size']), 0, collate_fn=lambda x: tuple(zip(*x)))
        valid_loader = get_dataloader(validset, 1, 0, collate_fn=lambda x: tuple(zip(*x)), shuffle=False)
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        scaler = torch.cuda.amp.GradScaler(enabled = False if device == 'cpu' else True)
        model.to(device)

        # construct an optimizer
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.001,
                                    momentum=0.9, weight_decay=0.0005)
        # and a learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=3,
                                                    gamma=0.1)
        num_epochs = int(self.cfg['settings']['epochs'])
        best_valid_loss = 10000000.0
        def evaluate(model, valid_loader):
            valid_bar = tqdm(valid_loader)
            loss = 0.0
            with torch.no_grad():
                for cnt, (images,targets) in enumerate(valid_bar):
                    images = list(image.to(device) for image in images)
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                    with torch.cuda.amp.autocast():
                        loss_dict = model(images, targets)
                    loss = sum(loss for loss in loss_dict.values())
                    loss += loss.item()
                    valid_bar.set_description(f"Evaluating: Loss: {loss/(cnt+1)}")
            model.zero_grad()
            return loss / len(valid_bar)
               
        #######TRAIN
        for i in range(num_epochs):
            loss = 0.0
            training_bar = tqdm(loader)
            for cnt, (images,targets) in enumerate(training_bar):
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                with torch.cuda.amp.autocast():
                    loss_dict = model(images, targets)
                loss = sum(loss for loss in loss_dict.values())
                loss += loss.item()
                training_bar.set_description(f"Training: Epoch:{i} Loss: {loss/(cnt+1)} Best: {best_valid_loss}")
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                model.zero_grad()
            valid_loss = evaluate(model, valid_loader)
            #lr_scheduler.step()
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save({
                        'model_state_dict': model.state_dict(),
                        'custom_data': self.cfg['settings']['custom_data'],
                        'model': modelstring}, os.path.join(savefolder, "maskrcnn.pt"))
            torch.save({
                        'model_state_dict': model.state_dict(),
                        'custom_data': self.cfg['settings']['custom_data'],
                        'model': modelstring}, os.path.join(savefolder, "maskrcnnlast.pt"))



        
    def train(self):

        pass

    def test(self):
        pass

        