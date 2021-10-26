from numpy import e
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
from .dpt.models import DPTSegmentationModel
from .custom import U2NET
from .TransInUnet import TransInUnet
from .hrnet import HRNetV2
#from .caranet.CaraNet import caranet
#from .doubleunet.doubleunet import DoubleUnet
#from .medtransformer.lib.models.axialnet import MedT, axialunet, gated, logo


class SegmentationModel(ModelBase):
    def __init__(self, encoder_name, decoder_name, nc, in_channels, criterions, metrics, monitor_metric, optimizer, lr, accumulate_batch, tensorboard_dir, class_weights, calculate_weight_map, weight, save_folder, early_stopping, multi_scale = None, max_image_size = None, use_attention = False, custom_data = {}, calculate_distance_maps = False):
        modelstring = ""
        nc = nc if not calculate_distance_maps else nc+1
        if decoder_name.lower() == "U2NET".lower():
            model = U2NET(in_channels, nc)
            modelstring = f"U2NET({in_channels}, {nc})"
        elif decoder_name.lower() == "TransInUnet".lower():
            model = TransInUnet(max_image_size, nc)
            modelstring = f"TransInUnet({max_image_size}, {nc})"
        elif decoder_name.lower() == "HrNetV2".lower():
            model = HRNetV2(nc)
            modelstring = f"HRNetV2({nc})"
        elif decoder_name.lower() == "dpt":
            model = DPTSegmentationModel(nc, backbone = "vitb_rn50_384")
            modelstring = f'DPTSegmentationModel({nc}, backbone = "vitb_rn50_384")'
        # elif decoder_name.lower() == "caranet":
        #     model = caranet(num_classes=nc)
        #     modelstring = f'caranet(num_classes = {nc})'
        # elif decoder_name.lower() == "doubleunet":
        #     model = DoubleUnet(encoder_name=encoder_name,classes=nc)
        #     modelstring = f'DoubleUnet(encoder_name = "{encoder_name}", classes = {nc})'
        # elif decoder_name.lower() == "medicaltransformer":
        #     model = MedT(img_size = max_image_size, imgchan = 3, num_classes = nc)
        #     modelstring = f'MedT(img_size = {max_image_size}, imgchan = {3}, num_classes = {nc})'
        else:
            if use_attention:
                model = eval(f'smp.create_model(arch="{decoder_name}", encoder_name="{encoder_name}", encoder_weights="imagenet", in_channels={in_channels}, classes = {nc}, image_size = {max_image_size}, decoder_attention_type = "{"scse"}")')
                modelstring = f'smp.create_model(arch="{decoder_name}", encoder_name="{encoder_name}", encoder_weights="imagenet", in_channels={in_channels}, classes = {nc}, image_size = {max_image_size}, decoder_attention_type = "{"scse"}")'
            else:
                model = eval(f'smp.create_model(arch="{decoder_name}", encoder_name="{encoder_name}", encoder_weights="imagenet", in_channels={in_channels}, classes = {nc}, image_size = {max_image_size})')
                modelstring = f'smp.create_model(arch="{decoder_name}", encoder_name="{encoder_name}", encoder_weights="imagenet", in_channels={in_channels}, classes = {nc}, image_size = {max_image_size})'

        self.modelstring = modelstring

    
          
        super().__init__(nc, criterions, metrics, monitor_metric, optimizer, lr, accumulate_batch, tensorboard_dir,
         class_weights, model = model, calculate_weight_map = calculate_weight_map,
          weight = weight, save_folder=save_folder, early_stopping=early_stopping, custom_data = custom_data, calculate_distance_maps=calculate_distance_maps)
        self.encoder_name = encoder_name
        self.decoder_name = decoder_name
        self.name = f"{encoder_name} - {decoder_name}"

    def __call__(self, x):
        return self.model(x)
    
    @staticmethod
    def create_model(modelstring):
        return eval(modelstring)

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
        early_stopping=self.early_stopping,
        modelstring=self.modelstring,
        custom_data=self.custom_data,
        distance_map_loss=self.distance_map_loss)

    def test(self, test_loader):
        pass
        
    def visualize(self, image):
        pass