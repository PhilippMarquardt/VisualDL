import timm
import yaml
from ...utils.utils import *
from ...models.classification_model import ClassificationModel
from ..trainer_base import TrainerBase
from ...utils.datasets import *
from ...utils.utils import *
import logging
import pprint
logging.getLogger().setLevel(logging.INFO)

class ClassificationTrainer(TrainerBase):
    def __init__(self, cfg_path):
        self.cfg = parse_yaml(cfg_path)
        logging.info(f"Training classification model with the following config:\n {pprint.pformat(self.cfg)}")
        assert self.cfg['type'] == "classification", "Provided yaml file must be a classification model config!"
        assert len(self.cfg['settings']['batch_size']) == len(self.cfg['model_names'])
        self.batch_sizes = self.cfg['settings']['batch_size']
        self.models = [ClassificationModel(name, self.cfg['settings']['nc'], self.cfg['settings']['criterions']) for name in self.cfg['model_names']]
        self.train_loaders = [get_dataloader(ClassificationDataset(self.cfg['data']['train'], transform = None), batch_size, self.cfg['settings']['workers']) for batch_size in self.batch_sizes]
        self.valid_loaders = [None] * len(self.batch_sizes)
        self.test_laoders = [None] * len(self.batch_sizes)
        if self.cfg['data']['valid'] != '':
            self.valid_loaders = [get_dataloader(ClassificationDataset(self.cfg['data']['valid'], transform = None), batch_size, self.cfg['settings']['workers']) for batch_size in self.batch_sizes]
        if self.cfg['data']['test'] != '':
            self.valid_loaders = [get_dataloader(ClassificationDataset(self.cfg['data']['test'], transform = None), batch_size, self.cfg['settings']['workers']) for batch_size in self.batch_sizes]
        
    def train(self):
        for cnt, (train_loader, valid_loader, test_loader, model) in enumerate(zip(self.train_loaders, self.valid_loaders, self.test_laoders, self.models)):
            logging.info(f"Training {pprint.pformat(model.__name__)}")
            model.train(train_loader, valid_loader, test_loader, self.cfg['settings']['epochs']) 
        


