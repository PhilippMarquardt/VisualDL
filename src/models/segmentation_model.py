from src.models.model_base import ModelBase
import segmentation_models_pytorch as smp
from torch.utils.data.dataloader import DataLoader
from torch.nn import *
from torch.optim import *
from torch.utils.tensorboard import SummaryWriter

class SegmentationModel(ModelBase):
    def __init__(self, encoder_name, decoder_name,nc, in_channels, criterions, metrics, montitor_metric, optimizer, lr, accumulate_batch, tensorboard_dir):
        self.encoder_name = encoder_name
        self.decoder_name = decoder_name
        self.model = eval(f'smp.{decoder_name}(encoder_name="{encoder_name}", encoder_weights="imagenet", in_channels={in_channels}, classes = {nc})')

        self.writer = SummaryWriter(tensorboard_dir)
        self.optimizer = eval(f"{optimizer}(self.model.parameters(), lr={lr})")
        self.transform = None #TODO: Read transform from config
        self.criterions = [eval(name)() for name in criterions]
        self.metrics = metrics
        self.monitor_metric = montitor_metric
        self.accumulate_batch = accumulate_batch

    def __call__(self, x):
        return self.model(x)

    def train(self, train_loader:DataLoader, valid_loader:DataLoader = None, test_loader:DataLoader = None, epochs:int = 1):
        """Trains the model on the given DataLoader.

        Args:
            train_loader (DataLoader): The training DataLoader
            valid_loader (DataLoader, optional): The valid DataLoader. Defaults to None.
            test_loader (DataLoader, optional): The test DataLoader. Defaults to None.
            epochs (int, optional): The number of epochs. Defaults to 1.
        """
        pass

    def test(self, test_loader):
        pass
        
    def visualize(self, image):
        pass