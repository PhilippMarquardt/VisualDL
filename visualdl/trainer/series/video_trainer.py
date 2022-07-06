from turtle import forward
import timm
import yaml
from ...utils.utils import *
from ..trainer_base import TrainerBase
from ...utils.datasets import *
from ...utils.utils import *
from torch import nn
import logging
import pprint
from torchmetrics import * 
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from ...utils.model_utils import test_trainer
logging.getLogger().setLevel(logging.INFO)
from torch.optim import *

class Classifier(nn.Module):
    def __init__(self, encoder, num_classes = 3, hidden = 256):
        super().__init__()
        num_out_channels = encoder.feature_info.channels()[-1]
        self.lstm = torch.nn.LSTM(input_size = num_out_channels, hidden_size = hidden, num_layers = 1, batch_first = True, bidirectional = True)
        self.out = nn.Linear(hidden * 2, num_classes)
        self.encoder = encoder
        self.flatten = torch.nn.Flatten()
    def forward(self, x):
        inp = []
        for point in x.swapaxes(0,1):
            out = self.flatten(F.adaptive_avg_pool2d(self.encoder(point)[-1], (1, 1)))
            inp.append(out)
        out,(ht,ct) = self.lstm(torch.stack(inp, dim=1))
        x = self.out(torch.cat([ht[0],ht[-1]],dim=1))
        print(x.shape)
        return x


class VideoTrainer():
    def __init__(self, cfg_path:dict):
        self.cfg = parse_yaml(cfg_path)
        assert self.cfg['type'] == "video", "Provided yaml file must be a classification model config!"
        self.encoder = timm.create_model(self.cfg['settings']['modelname'], pretrained=True if is_internet_connection_availible() else False, features_only=True)
        self.model = Classifier(self.encoder)                                   
                                            
    def train(self):    
        train_x = torch.rand((200, 10, 3, 128, 128), dtype = torch.float) #200 videos, 10 bilder pro video, 3 channel, 128 width, 128 height
        train_y = torch.tensor([1] * 200, dtype = torch.long)
        batch_size = 4
        for i in range(0, train_x.shape[0] // batch_size, batch_size):
            
            self.model(train_x[i:i+batch_size])
    
    def test(self):
        assert self.trained, "Must be trained first!"
        return test_trainer(self.models, self.test_loaders, [self.monitor_metric])



