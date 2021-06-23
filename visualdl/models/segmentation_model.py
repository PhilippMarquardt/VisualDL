from src.models.model_base import ModelBase
from src.utils.model_utils import *
import segmentation_models_pytorch as smp
from torch.utils.data.dataloader import DataLoader
from ..utils.losses import *
from torch.nn import *
from torch.optim import *
from uformer_pytorch import Uformer
from torch.utils.tensorboard import SummaryWriter

class SegmentationModel(ModelBase):
    def __init__(self, encoder_name, decoder_name, nc, in_channels, criterions, metrics, optimizer, lr, accumulate_batch, tensorboard_dir):
        self.encoder_name = encoder_name
        self.decoder_name = decoder_name
        print(f'smp.{decoder_name}(encoder_name="{encoder_name}", encoder_weights="imagenet", in_channels={in_channels}, classes = {nc})')

        self.model = eval(f'smp.{decoder_name}(encoder_name="{encoder_name}", encoder_weights="imagenet", in_channels={in_channels}, classes = {nc})')

        self.writer = SummaryWriter(tensorboard_dir)
        self.optimizer = eval(f"{optimizer}(self.model.parameters(), lr={lr})")
        self.bb = smp.losses.DiceLoss(mode="multiclass")
        self.criterions = [eval(f"{name}(reduction='none')") if name not in ["DiceLoss"] else eval(f"{name}(mode='multiclass')") for name in criterions]
        self.metrics = metrics
        self.accumulate_batch = accumulate_batch
        self.name = f"{encoder_name} - {decoder_name}"

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
        train_all_epochs(model = self.model, train_loader = train_loader, valid_loader=valid_loader, test_loader = test_loader, 
        epochs=epochs, criterions=self.criterions, metrics = self.metrics, writer=self.writer, name=f"{self.encoder_name}, {self.decoder_name}", optimizer=self.optimizer, accumulate_batch=self.accumulate_batch)

    def test(self, test_loader):
        pass
        
    def visualize(self, image):
        pass