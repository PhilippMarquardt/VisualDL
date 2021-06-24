import timm
from torch.utils.data.dataloader import DataLoader
from .model_base import ModelBase
from ..utils.utils import get_dataloader, write_image
import albumentations as A
from torch.nn import *
import logging
import torch
from skimage import io
import numpy as np
import cv2
from ..utils.model_utils import evaluate, visualize,  train_all_epochs
from torch.utils.tensorboard import SummaryWriter
from torch.optim import *

class ClassificationModel(ModelBase):
    def __init__(self, name, nc, criterions, metrics, monitor_metric, optimizer, lr, accumulate_batch, tensorboard_dir):
        self.writer = SummaryWriter(tensorboard_dir)
        self.name = name
        self.model = timm.create_model(name, pretrained=True, num_classes = nc)
        self.optimizer = eval(f"{optimizer}(self.model.parameters(), lr={lr})")
        self.criterions = [eval(f"{name}(reduction='none')") for name in criterions]
        self.metrics = metrics
        self.accumulate_batch = accumulate_batch
        self.monitor_metric = monitor_metric

    def __call__(self, x):
        return self.model(x)

    def train(self, train_loader:DataLoader, valid_loader:DataLoader = None, test_loader:DataLoader = None, epochs:int = 1):
        """Trains the model on the given DataLoader.

        Args:
            train_loader (DataLoader): The training DataLoader
            valid_loader (DataLoader, optional): The valid DataLoader. Defaults to None.
            test_loader (DataLoader, optional): The test DataLoader. Defaults to None.
            epochs (int, optional): The number of epochs. Defaults to 1.
        """
        #if hasattr(train_loader.dataset, 'class_weights'):
        #    for crit in self.criterions:
        #        if crit.weight is None:
        #            crit.weight = torch.tensor(train_loader.dataset.class_weights).to('cuda:0' if torch.cuda.is_available() else 'cpu')
        train_all_epochs(model = self.model, train_loader = train_loader, valid_loader=valid_loader, test_loader = test_loader, 
        epochs=epochs, criterions=self.criterions, metrics = self.metrics, monitor_metric = self.monitor_metric, writer=self.writer, name=self.name, optimizer=self.optimizer, accumulate_batch=self.accumulate_batch)

    def test(self, test_loader):
        pass
        
    def visualize(self, image):
        return visualize(self.model, self.model.layer4[-1], image)

        
        
        
    