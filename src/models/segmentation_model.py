from src.models.model_base import ModelBase
from ..dependencies import segmentation_modelspytorch

class SegmentationModel(ModelBase):
    def __init__(self, encoder_name,decoder_name ,nc, criterions, metrics, montitor_metric, optimizer, lr, accumulate_batch, tensorboard_dir):
        