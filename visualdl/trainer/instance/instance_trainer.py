from torch.nn.parameter import Parameter

# from torchvision.ops.misc import Conv2d
from torchvision.models.detection.anchor_utils import AnchorGenerator
from ...utils.datasets import InstanceSegmentationDataset
from ...utils.utils import *
from ...utils.utils import parse_yaml, get_transform_from_config, get_dataloader
import torchvision
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from tqdm import tqdm
import os
import sysconfig
import sys
import subprocess
import argparse

# from koila import LazyTensor, lazy
from .engine import train_one_epoch, evaluate

from .utils import GeneralizedRCNNTransform


class InstanceTrainer:
    def __init__(self, cfg_path: dict):
        self.cfg = parse_yaml(cfg_path)
        train_trans, val_trans = get_transform_from_config(cfg=self.cfg)
        trainset = InstanceSegmentationDataset(
            folder=self.cfg["data"]["train"], transform=train_trans
        )
        validset = InstanceSegmentationDataset(
            folder=self.cfg["data"]["valid"], transform=val_trans
        )
        self.savefolder = self.cfg["data"]["save_folder"]
        weight_path = self.cfg["data"]["weights"]
        # load a model pre-trained on COCO

        # self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True, box_detections_per_img = self.cfg['settings']['max_boxes_per_image'], min_size = self.cfg['settings']['image_size'], rpn_anchor_generator = AnchorGenerator(
        #                ((8,), (16,), (32,), (64,), (128,)), ((0.5, 1.0, 2.0),) * 5
        #            ))

        is_internet = is_internet_connection_availible()
        if "nc" in self.cfg["settings"].keys():
            self.nc = self.cfg["settings"]["nc"]
        else:
            self.nc = 91  # 91 is standard classes imagenet

        # self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True,
        # box_detections_per_img = self.cfg['settings']['max_boxes_per_image'],
        # min_size = self.cfg['settings']['image_size'],
        # num_classes=self.nc)

        try:
            print("Trying to load pretrained maskrcnn_resnet50_fpn")
            model = torchvision.models.detection.maskrcnn_resnet50_fpn(
                pretrained=True,
                box_detections_per_img=self.cfg["settings"]["max_boxes_per_image"],
                min_size=self.cfg["settings"]["image_size"],
            )
            print("Loaded pretrained maskrcnn_resnet50_fpn")
        except:
            print("Could not load pretrained maskrcnn_resnet50_fpn")
            print("initializing maskrcnn_resnet50_fpn randomly")
            model = torchvision.models.detection.maskrcnn_resnet50_fpn(
                pretrained=False,
                pretrained_backbone=False,
                box_detections_per_img=self.cfg["settings"]["max_boxes_per_image"],
                min_size=self.cfg["settings"]["image_size"],
            )
        if self.cfg["settings"]["in_channels"] != 3:
            model.transform = GeneralizedRCNNTransform(
                image_mean=[0.485, 0.456, 0.406],
                image_std=[0.229, 0.224, 0.225],
                min_size=(800,),
                max_size=1333,
                mode="bilinear",
            )
        model.backbone.body.conv1 = torch.nn.Conv2d(
            self.cfg["settings"]["in_channels"],
            model.backbone.body.conv1.out_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )

        # model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True,
        # box_detections_per_img = self.cfg['settings']['max_boxes_per_image'],
        # min_size = self.cfg['settings']['image_size'], trainable_backbone_layers=0,
        # rpn_anchor_generator = AnchorGenerator(
        #                ((16,), (32,), (64,), (128,), (256,)), ((0.5, 1.0, 2.0),) * 5
        #         ))

        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.nc)

        # now get the number of input features for the mask classifier
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        # and replace the mask predictor with a new one
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask, hidden_layer, self.nc
        )
        self.model = model
        # print(self.model)
        # self.modelstring = f"torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained={True}, box_detections_per_img = {self.cfg['settings']['max_boxes_per_image']}, min_size =  {self.cfg['settings']['image_size']}, rpn_anchor_generator = {AnchorGenerator(((8,), (16,), (32,), (64,), (128,)), ((0.5, 1.0, 2.0),) * 5)}"
        self.modelstring = f"torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained={False},\
                 pretrained_backbone={False},\
                 box_detections_per_img = {self.cfg['settings']['max_boxes_per_image']},\
                 min_size = {self.cfg['settings']['image_size']},\
                 num_classes={self.nc})"

        # self.modelstring = f"torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained={False},\
        #          box_detections_per_img = {self.cfg['settings']['max_boxes_per_image']},\
        #          min_size = {self.cfg['settings']['image_size']},\
        #          num_classes={self.nc},\
        #          rpn_anchor_generator = AnchorGenerator(\
        #                ((16,), (32,), (64,), (128,), (256,)), ((0.5, 1.0, 2.0),) * 5\
        #         ))"
        if os.path.isfile(weight_path):
            if weight_path != "None" and weight_path.endswith(".pt"):
                try:
                    own_state = self.model.state_dict()
                    load_state = torch.load(weight_path)["model_state_dict"]
                    for name, param in load_state.items():
                        if name not in own_state:
                            continue
                        else:
                            if own_state[name].shape != load_state[name].shape:
                                continue
                        if isinstance(param, Parameter):
                            # backwards compatibility for serialized parameters
                            param = param.data
                        own_state[name].copy_(param)

                    # self.model.load_state_dict(torch.load(weight)['model_state_dict'], strict = False)
                except Exception as e:
                    print(e)
                    logging.warning(f"Could not load weights from {weight_path}")
            # try:
            #     self.model.load_state_dict(torch.load(self.cfg['data']['weights'])['model_state_dict'], strict=False)
            #     print("Weights loaded!")
            # except:
            #     print("Could not load weights!")

        self.loader = get_dataloader(
            trainset,
            int(self.cfg["settings"]["batch_size"]),
            0,
            collate_fn=lambda x: tuple(zip(*x)),
        )
        self.valid_loader = get_dataloader(
            validset, 1, 0, collate_fn=lambda x: tuple(zip(*x)), shuffle=False
        )
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def train(self):
        scaler = torch.cuda.amp.GradScaler(
            enabled=False if self.device == "cpu" else True
        )
        self.model.to(self.device)

        # construct an optimizer
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(
            params, lr=0.0001, momentum=0.9, weight_decay=0.0005
        )

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=1e-4, weight_decay=1e-5
        )
        # and a learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=3, gamma=0.1
        )
        num_epochs = int(self.cfg["settings"]["epochs"])
        best_valid_loss = 10000000.0

        def evaluate(model, valid_loader):
            valid_bar = tqdm(valid_loader, file=sys.stdout)
            total_loss = 0.0
            with torch.no_grad():
                for cnt, (images, targets) in enumerate(valid_bar):
                    images = list(image.to(self.device) for image in images)
                    targets = [
                        {k: v.to(self.device) for k, v in t.items()} for t in targets
                    ]
                    with torch.cuda.amp.autocast():
                        loss_dict = model(images, targets)
                        loss = sum(loss for loss in loss_dict.values())
                        total_loss += loss.detach().item()
                    valid_bar.set_description(f"Evaluating: Loss: {total_loss/(cnt+1)}")
            model.zero_grad()
            return total_loss / len(valid_bar)

        #######TRAIN
        for i in range(num_epochs):
            # train_one_epoch(self.model, optimizer, self.loader, self.device, i, print_freq=10, scaler=scaler)
            # evaluate(self.model, self.valid_loader, device=self.device)
            total_loss = 0.0
            training_bar = tqdm(self.loader, file=sys.stdout)
            for cnt, (images, targets) in enumerate(training_bar):
                images = list(image.to(self.device) for image in images)
                targets = [
                    {k: v.to(self.device) for k, v in t.items()} for t in targets
                ]
                # (images, targets) = lazy(images, targets, batch=0)
                with torch.cuda.amp.autocast():
                    loss_dict = self.model(images, targets)
                    loss = sum(loss for loss in loss_dict.values())
                    total_loss += loss.detach().item()
                training_bar.set_description(
                    f"Training: Epoch:{i} Loss: {total_loss/(cnt+1)} Best: {best_valid_loss}"
                )
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                self.model.zero_grad()
            valid_loss = evaluate(self.model, self.valid_loader)
            # lr_scheduler.step()
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "custom_data": self.cfg["settings"]["custom_data"],
                        "nc": self.cfg["settings"]["nc"],
                        "model": self.modelstring,
                        "in_channels": self.cfg["settings"]["in_channels"],
                        "image_size": 512,
                    },
                    os.path.join(self.savefolder, "maskrcnn.pt"),
                )
            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "custom_data": self.cfg["settings"]["custom_data"],
                    "nc": self.cfg["settings"]["nc"],
                    "model": self.modelstring,
                    "in_channels": self.cfg["settings"]["in_channels"],
                    "image_size": 512,
                },
                os.path.join(self.savefolder, "maskrcnnlast.pt"),
            )
        # pass

    def test(self):
        pass
