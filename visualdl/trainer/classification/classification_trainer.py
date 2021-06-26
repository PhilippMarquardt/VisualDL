import timm
import yaml
from ...utils.utils import *
from ...models.classification_model import ClassificationModel
from ..trainer_base import TrainerBase
from ...utils.datasets import *
from ...utils.utils import *
import logging
import pprint
from torchmetrics import * 
import matplotlib.pyplot as plt
import numpy as np
from ...utils.model_utils import test_trainer
logging.getLogger().setLevel(logging.INFO)
from torch.optim import *


"""
    The container class around mutliple models responsible for training them all with the paramers specific in the config file.
"""
class ClassificationTrainer(TrainerBase):
    def __init__(self, cfg_path:dict):
        super().__init__(cfg_path)
        assert self.cfg['type'] == "classification", "Provided yaml file must be a classification model config!"
        assert len(self.cfg['settings']['batch_size']) == len(self.cfg['model_names'])    
        logging.info(f"Training classification model with the following config:\n {pprint.pformat(self.cfg)}")    
        self.models = [ClassificationModel(name, self.nc, self.critetions, self.metrics, self.monitor_metric,
                                            self.optimizer, self.lr, self.gradient_accumulation,self.tensorboard_dir, self.class_weights, self.calculate_weight_map) for name in self.cfg['model_names']]
                                            
    def train(self):
        logging.info("Starting training training!")
        for cnt, (train_loader, valid_loader, test_loader, model) in enumerate(zip(self.train_loaders, self.valid_loaders, self.test_laoders, self.models)):
            logging.info(f"Training {pprint.pformat(self.cfg['model_names'][cnt])}")
            model.train(train_loader, valid_loader, test_loader, self.epochs) 
        self.trained = True
    
    def test(self):
        assert self.trained, "Must be trained first!"
        return test_trainer(self.models, self.test_loaders, [self.monitor_metric])

    def get_visualization(self):
        for cnt, (x,y) in enumerate(self.train_loaders[0]):
            out = self.models[1].visualize(x[0,:])
            write_image(f"{cnt}xd.png", out) 
            #write_image("xd.png", self.models[1].visualize(2, x[0]))
        


