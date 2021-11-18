from abc import ABC, abstractmethod
from ..utils.utils import parse_yaml, get_transform_from_config, get_dataloader
from ..utils.datasets import SegmentationDataset, ClassificationDataset
from torch.nn import *
from torchmetrics import * 
import torch

class TrainerBase(ABC):
    def __init__(self, cfg):
        self.cfg = parse_yaml(cfg)
        self.trained = False
        # Parse all attributes that are shared across all trainers
        assert len(self.cfg['settings']['metrics']) > 0, "You must provide atleast one metric"
        assert all([self.cfg['settings']['gradient_accumulation'] % batch == 0 for batch in self.cfg['settings']['batch_size']])
        self.type = self.cfg["type"]
        self.batch_sizes = self.cfg['settings']['batch_size']
        self.nc = self.cfg['settings']['nc']
        self.critetions = self.cfg['settings']['criterions']
        self.metrics = [eval(f"{metric['name']} ({metric['params']})") for metric in self.cfg['settings']['metrics']]
        self.monitor_metric = eval(f"{self.cfg['settings']['monitor_metric_name']} ({self.cfg['settings']['monitor_metric_params']})")
        self.optimizer = self.cfg['settings']['optimizer']
        self.lr = self.cfg['settings']['lr']
        self.gradient_accumulation = self.cfg['settings']['gradient_accumulation']
        self.tensorboard_dir = self.cfg['settings']['tensorboard_log_dir']
        self.workers = self.cfg['settings']['workers']
        self.train_path = self.cfg['data']['train']
        self.valid_path = self.cfg['data']['valid']
        self.test_path = self.cfg['data']['test']
        self.epochs = self.cfg['settings']['epochs']
        self.weights = self.cfg['data']['weights']
        self.save_folder = self.cfg['data']['save_folder']
        self.early_stopping = self.cfg['settings']['early_stopping']
        self.calculate_class_weights = self.cfg['settings']['class_weights']
        self.use_attention = self.cfg['settings']['use_attention']
        self.custom_data = self.cfg['settings']['custom_data']
        transforms, valid_trans = get_transform_from_config(cfg=self.cfg)
        #initialize loaders
        if self.type == "segmentation": 
            dataset = SegmentationDataset
            self.calculate_weight_map = self.cfg['settings']['calculate_weight_map']
        elif self.type == "classification":
            dataset = ClassificationDataset
        self.train_loaders = [get_dataloader(dataset(self.train_path, transform = transforms, class_weights=self.calculate_class_weights), batch_size, self.workers) for batch_size in self.batch_sizes]
        self.valid_loaders = [None] * len(self.batch_sizes)
        self.test_laoders = [None] * len(self.batch_sizes)
        if self.valid_path != '':
            self.valid_loaders = [get_dataloader(dataset(self.valid_path, transform = valid_trans), batch_size, 0) for batch_size in self.batch_sizes]
        if self.test_path != '':
            self.test_loaders = [get_dataloader(dataset(self.test_path, transform = valid_trans), batch_size, 0) for batch_size in self.batch_sizes]
        self.class_weights = torch.FloatTensor(self.train_loaders[0].dataset.class_weights).to('cuda:0' if torch.cuda.is_available() else 'cpu') if self.calculate_class_weights else None
        self.stopped = False
    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def test(self):
        pass

    def stop(self):
        self.stopped = True
