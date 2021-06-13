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
        self.cfg = parse_yaml(cfg_path)
        self.trained = False
        assert self.cfg['type'] == "classification", "Provided yaml file must be a classification model config!"
        assert len(self.cfg['settings']['batch_size']) == len(self.cfg['model_names'])
        assert len(self.cfg['settings']['metrics']) > 0, "You must provide atleast one metric"
        assert all([self.cfg['settings']['gradient_accumulation'] % batch == 0 for batch in self.cfg['settings']['batch_size']])
        logging.info(f"Training classification model with the following config:\n {pprint.pformat(self.cfg)}")
        

        self.batch_sizes = self.cfg['settings']['batch_size']
        self.models = [ClassificationModel(name, self.cfg['settings']['nc'], self.cfg['settings']['criterions'],
                      [eval(f"{metric['name']} ({metric['params']})") for metric in self.cfg['settings']['metrics']],
                       self.cfg['settings']['optimizer'], self.cfg['settings']['lr'],
                       self.cfg['settings']['gradient_accumulation'],
                       self.cfg['settings']['tensorboard_log_dir']) for name in self.cfg['model_names']]

        
        transforms = get_transform_from_config(cfg=self.cfg)
        self.train_loaders = [get_dataloader(ClassificationDataset(self.cfg['data']['train'], transform = transforms), batch_size, self.cfg['settings']['workers']) for batch_size in self.batch_sizes]
        self.valid_loaders = [None] * len(self.batch_sizes)
        self.test_laoders = [None] * len(self.batch_sizes)
        if self.cfg['data']['valid'] != '':
            self.valid_loaders = [get_dataloader(ClassificationDataset(self.cfg['data']['valid'], transform = transforms), batch_size, self.cfg['settings']['workers']) for batch_size in self.batch_sizes]
        if self.cfg['data']['test'] != '':
            self.test_loaders = [get_dataloader(ClassificationDataset(self.cfg['data']['test'], transform = transforms), batch_size, self.cfg['settings']['workers']) for batch_size in self.batch_sizes]
        

    def train(self):
        logging.info("Starting training training!")
        for cnt, (train_loader, valid_loader, test_loader, model) in enumerate(zip(self.train_loaders, self.valid_loaders, self.test_laoders, self.models)):
            logging.info(f"Training {pprint.pformat(self.cfg['model_names'][cnt])}")
            model.train(train_loader, valid_loader, test_loader, self.cfg['settings']['epochs']) 
        self.trained = True
    
    def test(self):
        assert self.trained, "Must be trained first!"
        return test_trainer(self.models, self.test_loaders, self.models[0].metrics[0])

    def get_visualization(self):
        for cnt, (x,y) in enumerate(self.train_loaders[0]):
            out = self.models[1].visualize(x[0,:])
            write_image(f"{cnt}xd.png", out) 
            #write_image("xd.png", self.models[1].visualize(2, x[0]))
        


