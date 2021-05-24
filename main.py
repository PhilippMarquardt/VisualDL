from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from src.vdl import *
from src.utils.datasets import *
from src.utils.utils import *
from src.models.classification_model import *
from src.trainer.classification.classification_trainer import *
import logging
logging.getLogger().setLevel(logging.INFO)
logging.getLogger().name = ""

#main()
#def main():
t = ClassificationTrainer(r"E:\source\repos\VisualDL\src\trainer\classification\classification.yaml")
t.train()
#t.get_visualization()
di = t.test()
print(di)