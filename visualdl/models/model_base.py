from abc import ABC, abstractmethod
from torch.optim import *
from torch.nn import *
from torch.utils.tensorboard import SummaryWriter

class ModelBase(ABC):    
    def __init__(self, nc, criterions, metrics, monitor_metric, optimizer, lr, accumulate_batch, tensorboard_dir, class_weights, model):
        self.nc = nc
        self.model = model
        self.criterions = criterions
        self.metrics = metrics
        self.lr = lr
        self.criterions = [eval(f"{name}(reduction='none')") for name in criterions]
        self.writer = SummaryWriter(tensorboard_dir)
        self.monitor_metric = monitor_metric
        self.accumulate_batch = accumulate_batch
        self.optimizer = eval(f"{optimizer}(self.model.parameters(), lr={self.lr})")
        if class_weights is not None:
            for crit in self.criterions:
                crit.weight = class_weights

        
        
    @abstractmethod
    def train(self):
        pass
    
    @abstractmethod
    def test(self):
        pass
     
    @abstractmethod     
    def visualize(self):
        pass