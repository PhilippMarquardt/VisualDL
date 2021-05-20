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
logging.getLogger().setLevel(logging.INFO)

"""
    The container class around mutliple models responsible for training them all with the paramers specific in the config file.
"""
class ClassificationTrainer(TrainerBase):
    def __init__(self, cfg_path:dict):
        self.cfg = parse_yaml(cfg_path)
        assert self.cfg['type'] == "classification", "Provided yaml file must be a classification model config!"
        assert len(self.cfg['settings']['batch_size']) == len(self.cfg['model_names'])
        assert len(self.cfg['settings']['metrics']) > 0, "You must provide atleast one metric"
        logging.info(f"Training classification model with the following config:\n {pprint.pformat(self.cfg)}")
        self.batch_sizes = self.cfg['settings']['batch_size']
        self.models = [ClassificationModel(name, self.cfg['settings']['nc'], self.cfg['settings']['criterions'],
                      [eval(f"{metric['name']} ({metric['params']})") for metric in self.cfg['settings']['metrics']]) for name in self.cfg['model_names']]

        self.train_loaders = [get_dataloader(ClassificationDataset(self.cfg['data']['train'], transform = None), batch_size, self.cfg['settings']['workers']) for batch_size in self.batch_sizes]
        self.valid_loaders = [None] * len(self.batch_sizes)
        self.test_laoders = [None] * len(self.batch_sizes)
        if self.cfg['data']['valid'] != '':
            self.valid_loaders = [get_dataloader(ClassificationDataset(self.cfg['data']['valid'], transform = None), batch_size, self.cfg['settings']['workers']) for batch_size in self.batch_sizes]
        if self.cfg['data']['test'] != '':
            self.valid_loaders = [get_dataloader(ClassificationDataset(self.cfg['data']['test'], transform = None), batch_size, self.cfg['settings']['workers']) for batch_size in self.batch_sizes]
        

    def train(self):
        for cnt, (train_loader, valid_loader, test_loader, model) in enumerate(zip(self.train_loaders, self.valid_loaders, self.test_laoders, self.models)):
            logging.info(f"Training {pprint.pformat(self.cfg['model_names'][cnt])}")
            model.train(train_loader, valid_loader, test_loader, self.cfg['settings']['epochs']) 
        


