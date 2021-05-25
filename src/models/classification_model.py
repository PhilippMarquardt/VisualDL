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


class ClassificationModel(ModelBase):
    def __init__(self, name, nc, criterions, metrics, montitor_metric):
        self.writer = SummaryWriter(r"E:\source\repos\VisualDL\tensorboard_log")
        self.name = name
        self.model = timm.create_model(name, pretrained=True, num_classes = nc)
       
        
        self.transform = None #TODO: Read transform from config
        self.criterions = [eval(name)() for name in criterions]
        self.metrics = metrics
        self.monitor_metric = montitor_metric

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
        train_all_epochs(model = self.model, train_loader = train_loader, valid_loader=valid_loader, test_loader = test_loader, 
        epochs=epochs, criterions=self.criterions, metrics = self.metrics, writer=self.writer, name=self.name, monitor_metric = self.monitor_metric)

    def test(self, test_loader):
        pass
        
    def visualize(self, image):
        return visualize(self.model, self.model.layer4[-1], image)

        
        
        
    