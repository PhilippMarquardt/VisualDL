from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from src.vdl import *
from src.utils.datasets import *
from src.utils.utils import *
from src.models.classification_model import *
#main()
#def main():

a =ClassificationModel()
a.train()