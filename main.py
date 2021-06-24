from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from visualdl.vdl import *
from visualdl.utils.datasets import *
from visualdl.utils.utils import *
from visualdl.models.classification_model import *
from visualdl.trainer.classification.classification_trainer import *
from visualdl.trainer.segmentation.segmentation_trainer import *
import logging


from visualdl.models.segmentation_model import *


logging.getLogger().setLevel(logging.INFO)
logging.getLogger().name = ""


#bub = {'type': 'classification', 'data': {'train': 'C:/Users/phili/Downloads/Telegram Desktop/dataset', 'valid': 'C:/Users/phili/Downloads/Telegram Desktop/dataset_valid', 'test': 'C:/Users/phili/Downloads/Telegram Desktop/dataset_valid'}, 'model_names': ['regnety_004', 'efficientnet_b0', 'resnext50_32x4d'], 'settings': {'nc': 5, 'epochs': 100, 'optimizer': 'AdamW', 'lr': 0.0001, 'workers': 0, 'batch_size': [16, 16, 8], 'gradient_accumulation': 16, 'criterions': ['CrossEntropyLoss'], 'tensorboard_log_dir': 'tensorboard_logs', 'metrics': [{'name': 'F1', 'params': 'num_classes=5'}, {'name': 'Accuracy', 'params': ''}, {'name': 'CohenKappa', 'params': 'num_classes=5'}], 'monitor_metric_name': 'Accuracy', 'monitor_metric_params': 'num_classes=5'}, 'transforms': {'width': 512, 'height': 512, 'h_flip': 0.5, 'v_flip': 0.5, 'brightness': 0.0, 'contrast': 0.0, 'rgb_shift': 0.0, 'random_shadow': 0.0, 'blur': 0.0}}

#main()
#def main():
#t = ClassificationTrainer(bub)
#t.train()
#t.get_visualization()
#di = t.test()
#print(di)
#t = SegmentationModel("resnet34", "Unet", 3, 3, None, None, None, None, None, None, None)
t = SegmentationTrainer(r"E:\source\repos\VisualDL\visualdl\trainer\segmentation\segmentation.yaml")
t.train()
#print(t.test())