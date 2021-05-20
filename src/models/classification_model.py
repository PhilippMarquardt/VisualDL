import timm
from .model_base import ModelBase
from ..utils.utils import train_all_epochs
import albumentations as A
from torch.nn import *
import logging
#logging.getLogger().setLevel(logging.INFO)
logging.getLogger().name = "Classification Model"

class ClassificationModel(ModelBase):
    def __init__(self, name, nc, criterions):
        self.model = timm.create_model(name, pretrained=True)
        self.model.fc = Linear(self.model.fc.in_features, nc)
        logging.info("XD")
        self.transform = None #TODO: Read transform from config
        self.criterions = [eval(name)() for name in criterions]
        #self.criterions = [CrossEntropyLoss]
    def train(self, train_loader, valid_loader = None, test_loader = None, epochs = 1):
        train_all_epochs(model = self.model, train_loader = train_loader, valid_loader=valid_loader, test_loader = test_loader, epochs=epochs, criterions=self.criterions)
        
    def test(self):
        pass
        
    def inference(self):
        pass
        
    def visualize(self):
        pass
        
        
        
    