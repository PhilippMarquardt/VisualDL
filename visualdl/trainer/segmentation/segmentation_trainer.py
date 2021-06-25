import timm
import yaml
from ...utils.utils import *
from ...models.segmentation_model import SegmentationModel
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
class SegmentationTrainer(TrainerBase):
    def __init__(self, cfg_path:dict):
        super().__init__(cfg_path)
        assert self.cfg['type'] == "segmentation", "Provided yaml file must be a segmentation model config!"
        assert len(self.cfg['settings']['batch_size']) == len(self.cfg['models'])
        logging.info(f"Training segmentation model with the following config:\n {pprint.pformat(self.cfg)}")
        


        self.models = [SegmentationModel(models['backbone'],
                        models['decoder'], 
                        self.nc, 
                        self.cfg['settings']['in_channels'],
                        self.critetions,
                        self.metrics,
                        self.monitor_metric,
                        self.optimizer, 
                        self.lr,
                        self.gradient_accumulation,
                        self.tensorboard_dir)  for models in self.cfg['models']]

        
        transforms = get_transform_from_config(cfg=self.cfg)
        self.train_loaders = [get_dataloader(SegmentationDataset(self.cfg['data']['train'], transform = transforms), batch_size, self.cfg['settings']['workers']) for batch_size in self.batch_sizes]
        self.valid_loaders = [None] * len(self.batch_sizes)
        self.test_laoders = [None] * len(self.batch_sizes)
        if self.cfg['data']['valid'] != '':
            self.valid_loaders = [get_dataloader(SegmentationDataset(self.cfg['data']['valid'], transform = transforms), batch_size, self.cfg['settings']['workers']) for batch_size in self.batch_sizes]
        if self.cfg['data']['test'] != '':
            self.test_loaders = [get_dataloader(SegmentationDataset(self.cfg['data']['test'], transform = transforms), batch_size, self.cfg['settings']['workers']) for batch_size in self.batch_sizes]
        

    def train(self):
        logging.info("Starting training training!")
        for cnt, (train_loader, valid_loader, test_loader, model) in enumerate(zip(self.train_loaders, self.valid_loaders, self.test_laoders, self.models)):
            logging.info(f"Training {model.name}")
            model.train(train_loader, valid_loader, test_loader, self.cfg['settings']['epochs']) 
        self.trained = True
    
    def test(self):
        assert self.trained, "Must be trained first!"
        return test_trainer(self.models, self.test_loaders, self.models[0].metrics)

    def get_visualization(self):
        for cnt, (x,y) in enumerate(self.train_loaders[0]):
            out = self.models[1].visualize(x[0,:])
            write_image(f"{cnt}xd.png", out) 
            #write_image("xd.png", self.models[1].visualize(2, x[0]))
        


