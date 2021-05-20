import timm
from torch.utils.data.dataloader import DataLoader
from .model_base import ModelBase
from ..utils.utils import get_dataloader, train_all_epochs
import albumentations as A
from torch.nn import *
import logging


class ClassificationModel(ModelBase):
    def __init__(self, name, nc, criterions, metrics):
        self.model = timm.create_model(name, pretrained=True)
        self.model.fc = Linear(self.model.fc.in_features, nc)
        self.transform = None #TODO: Read transform from config
        self.criterions = [eval(name)() for name in criterions]
        self.metrics = metrics
        #self.criterions = [CrossEntropyLoss]
    def train(self, train_loader:DataLoader, valid_loader:DataLoader = None, test_loader:DataLoader = None, epochs:int = 1):
        """Trains the model on the given DataLoader.

        Args:
            train_loader (DataLoader): The training DataLoader
            valid_loader (DataLoader, optional): The valid DataLoader. Defaults to None.
            test_loader (DataLoader, optional): The test DataLoader. Defaults to None.
            epochs (int, optional): The number of epochs. Defaults to 1.
        """
        train_all_epochs(model = self.model, train_loader = train_loader, valid_loader=valid_loader, test_loader = test_loader, epochs=epochs, criterions=self.criterions, metrics = self.metrics)
    def test(self):
        pass
        
    def inference(self):
        pass
        
    def visualize(self):
        pass
        
        
        
    