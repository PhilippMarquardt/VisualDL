from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from src.vdl import *
from src.utils.datasets import *
from src.utils.utils import *
from src.models.classification_model import *
from src.trainer.classification.classification_trainer import *
from src.trainer.segmentation.segmentation_trainer import *
import logging


from src.models.segmentation_model import *


logging.getLogger().setLevel(logging.INFO)
logging.getLogger().name = ""

#main()
#def main():
#t = ClassificationTrainer(r"E:\source\repos\VisualDL\src\trainer\classification\classification.yaml")
#t.train()
#t.get_visualization()
#di = t.test()
#print(di)
#t = SegmentationModel("resnet34", "Unet", 3, 3, None, None, None, None, None, None, None)
t = SegmentationTrainer(r"E:\source\repos\VisualDL\src\trainer\segmentation\segmentation.yaml")
t.train()
print(t.test())