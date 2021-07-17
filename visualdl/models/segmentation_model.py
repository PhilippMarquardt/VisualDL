from visualdl.utils.utils import timm_universal_encoders
from .model_base import ModelBase
from ..utils.model_utils import evaluate, visualize,  train_all_epochs
import segmentation_models_pytorch as smp
from torch.utils.data.dataloader import DataLoader
from ..utils.losses import *
from torch.nn import *
from torch.optim import *
from ..utils.utils import timm_universal_encoders
from .multiscale import MultiScaleSegmentation, HieraricalMultiScale
from uformer_pytorch import Uformer
from .custom import U2NET
from .TransInUnet import TransInUnet


class SegmentationModel(ModelBase):
    def __init__(self, encoder_name, decoder_name, nc, in_channels, criterions, metrics, monitor_metric, optimizer, lr, accumulate_batch, tensorboard_dir, class_weights, calculate_weight_map, weight, save_folder, early_stopping, multi_scale = None):
        #if encoder_name not in timm_universal_encoders(pretrained=True) and "timm-u" in encoder_name:
        #    model = eval(f'smp.{decoder_name}(encoder_name="{encoder_name}", encoder_weights="None", in_channels={in_channels}, classes = {nc})')
        #else:
        #try:
        
        if decoder_name == "U2NET":
            model = U2NET(in_channels, nc)
        elif decoder_name == "TransInUnet":
            model =TransInUnet(128, nc)
        else:
            #model = eval(f'smp.{decoder_name}(encoder_name="{encoder_name}", encoder_weights="imagenet", in_channels={in_channels}, classes = {nc}, image_size = 128)')
            model = eval(f'smp.create_model(arch="{decoder_name}", encoder_name="{encoder_name}", encoder_weights="imagenet", in_channels={in_channels}, classes = {nc}, image_size = {128})')
        #if multi_scale is not None and multi_scale is not 'None' and len(multi_scale) > 0:
        #    model = MultiScaleSegmentation(model, multi_scale)
        
        # except:
        #     model = Uformer(
        #             dim = 64,           # initial dimensions after input projection, which increases by 2x each stage
        #             stages = 4,         # number of stages
        #             num_blocks = 2,     # number of transformer blocks per stage
        #             window_size = 8,   # set window size (along one side) for which to do the attention within
        #             dim_head = 64,
        #             heads = 2,
        #             ff_mult = 2,
        #             output_channels=nc
        #         )
    
            
        super().__init__(nc, criterions, metrics, monitor_metric, optimizer, lr, accumulate_batch, tensorboard_dir, class_weights, model = model, calculate_weight_map = calculate_weight_map, weight = weight, save_folder=save_folder, early_stopping=early_stopping)
        self.encoder_name = encoder_name
        self.decoder_name = decoder_name
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
        epochs=epochs, criterions=self.loss, metrics = self.metrics, monitor_metric = self.monitor_metric, writer=self.writer, name=f"{self.encoder_name}, {self.decoder_name}", optimizer=self.optimizer, accumulate_batch=self.accumulate_batch,weight_map = self.calculate_weight_map,
        save_folder = self.save_folder,
        early_stopping=self.early_stopping)

    def test(self, test_loader):
        pass
        
    def visualize(self, image):
        pass