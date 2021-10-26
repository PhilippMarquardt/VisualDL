from abc import ABC, abstractmethod
from torch.optim import *
from torch.nn import *
from ..utils.losses import DiceLoss, MultiLoss
import torch
import logging
from torch.utils.tensorboard import SummaryWriter

class ModelBase(ABC):    
    def __init__(self, nc, criterions, metrics, monitor_metric, optimizer, lr, accumulate_batch, tensorboard_dir, class_weights, model, calculate_weight_map, weight, save_folder, early_stopping, custom_data, calculate_distance_maps):
        self.nc = nc
        self.model = model
        if weight != "None" and weight.endswith(".pt"):
            try:

                own_state = self.model.state_dict()
                load_state = torch.load(weight)['model_state_dict']
                for name, param in load_state.items():
                    if name not in own_state:
                        continue
                    else:
                        if own_state[name].shape != load_state[name].shape:
                            continue
                    if isinstance(param, Parameter):
                        # backwards compatibility for serialized parameters
                        param = param.data
                    own_state[name].copy_(param)
                

                #self.model.load_state_dict(torch.load(weight)['model_state_dict'], strict = False)
            except Exception as e:
                print(e)
                logging.warning(f"Could not load weights from {weight}")
        self.criterions = criterions
        self.metrics = metrics
        self.lr = lr  
        self.criterions = [eval(f"{name}(reduction='none')") for name in criterions]
        #self.criterions = [DiceLoss()]
        self.writer = SummaryWriter(tensorboard_dir)
        self.monitor_metric = monitor_metric
        self.accumulate_batch = accumulate_batch
        self.calculate_weight_map = calculate_weight_map
        self.early_stopping = early_stopping
        self.optimizer = eval(f"{optimizer}(self.model.parameters(), lr={self.lr})")
        if class_weights is not None:
            for crit in self.criterions:
                crit.weight = class_weights
        self.save_folder = save_folder
        self.loss = MultiLoss(self.criterions)
        self.custom_data = custom_data
        self.calculate_distance_maps = calculate_distance_maps
        self.distance_map_loss = None
        if self.calculate_distance_maps:
            self.distance_map_loss = torch.nn.MSELoss()
        
    
       
    @abstractmethod
    def train(self):
        pass
    
    @abstractmethod
    def test(self):
        pass
     
    @abstractmethod     
    def visualize(self):
        pass