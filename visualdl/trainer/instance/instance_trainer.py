from torchvision.ops.misc import Conv2d
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
from .engine import train_one_epoch, evaluate

class InstanceTrainer():
    def __init__(self, cfg_path:dict):
        self.cfg = parse_yaml(cfg_path)
        train_trans, val_trans = get_transform_from_config(cfg=self.cfg)
        trainset = InstanceSegmentationDataset(folder=self.cfg['data']['train'], transform = train_trans)
        validset = InstanceSegmentationDataset(folder=self.cfg['data']['valid'], transform = val_trans)
        self.savefolder = self.cfg['data']['save_folder']
        weight_path = self.cfg['data']['weights']
        # load a model pre-trained on COCO
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True, box_detections_per_img = self.cfg['settings']['max_boxes_per_image'])
        
        self.modelstring = f"torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained={True}, box_detections_per_img = {self.cfg['settings']['max_boxes_per_image']})"


        if "nc" in self.cfg['settings']:
            self.nc = self.cfg['settings']['nc']
            self.model.roi_heads.mask_predictor.mask_fcn_logits = Conv2d(256, self.cfg['settings']['nc'], 1)
        else:
            self.nc = 91 #91 is standard classes imagenet

        if os.path.isfile(weight_path):
            try:
                self.model.load_state_dict(torch.load(self.cfg['data']['weights'])['model_state_dict'], strict=False)
                print("Weights loaded!")
            except:
                print("Could not load weights!")
        
        
        
        self.loader = get_dataloader(trainset, int(self.cfg['settings']['batch_size']), 0, collate_fn=lambda x: tuple(zip(*x)))
        self.valid_loader = get_dataloader(validset, 1, 0, collate_fn=lambda x: tuple(zip(*x)), shuffle=False)
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        



        
    def train(self):
        scaler = torch.cuda.amp.GradScaler(enabled = False if self.device == 'cpu' else True)
        self.model.to(self.device)

        # construct an optimizer
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.0001,
                                    momentum=0.9, weight_decay=0.0005)
        # and a learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=3,
                                                    gamma=0.1)
        num_epochs = int(self.cfg['settings']['epochs'])
        best_valid_loss = 10000000.0
        def evaluate(model, valid_loader):
            valid_bar = tqdm(valid_loader, file=sys.stdout)
            loss = 0.0
            with torch.no_grad():
                for cnt, (images,targets) in enumerate(valid_bar):
                    images = list(image.to(self.device) for image in images)
                    targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                    with torch.cuda.amp.autocast():
                        loss_dict = model(images, targets)
                    loss = sum(loss for loss in loss_dict.values())
                    loss += loss.item()
                    valid_bar.set_description(f"Evaluating: Loss: {loss/(cnt+1)}")
            model.zero_grad()
            return loss / len(valid_bar)
               
        #######TRAIN
        for i in range(num_epochs):
            #train_one_epoch(self.model, optimizer, self.loader, self.device, i, print_freq=10, scaler=scaler)
            #evaluate(self.model, self.valid_loader, device=self.device)
            loss = 0.0
            training_bar = tqdm(self.loader, file=sys.stdout)
            for cnt, (images,targets) in enumerate(training_bar):
                images = list(image.to(self.device) for image in images)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                with torch.cuda.amp.autocast():
                    loss_dict = self.model(images, targets)
                loss = sum(loss for loss in loss_dict.values())
                loss += loss.item()
                training_bar.set_description(f"Training: Epoch:{i} Loss: {loss/(cnt+1)} Best: {best_valid_loss}")
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                self.model.zero_grad()
            valid_loss = evaluate(self.model, self.valid_loader)
            #lr_scheduler.step()
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'custom_data': self.cfg['settings']['custom_data'],
                        'nc': self.cfg['settings']['nc'],
                        'model': self.modelstring,
                        'image_size': 512}, os.path.join(self.savefolder, "maskrcnn.pt"))
            torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'custom_data': self.cfg['settings']['custom_data'],
                        'nc': self.cfg['settings']['nc'],
                        'model': self.modelstring,
                        'image_size': 512}, os.path.join(self.savefolder, "maskrcnnlast.pt"))
        # pass

    def test(self):
        pass

        