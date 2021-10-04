from .utils.utils import parse_yaml
from .trainer.segmentation.segmentation_trainer import SegmentationTrainer
from .trainer.classification.classification_trainer import ClassificationTrainer
from .trainer.detection.detection_trainer import DetectionTrainer
from .models.segmentation_model import SegmentationModel
from .utils.model_utils import predict_images
import torch
from torch import load
from .inference.inference import ModelInference
import sys

def train(cfg_path):
    type = parse_yaml(cfg_path)['type']
    if type == "classification":
        t = ClassificationTrainer(cfg_path=cfg_path)
    elif type == "segmentation":
        t = SegmentationTrainer(cfg_path=cfg_path)
    elif type == "od":
        not_unique_modules = ["models", "utils", "utils.metrics"]
        for module in not_unique_modules:            
            try:
                del sys.modules[module]
            except:
                pass
        t = DetectionTrainer(cfg_path=cfg_path)
    t.train()
    #print("DONE TRAINING")
    #print(t.test())


def predict(images, weights, device = 'cuda:0' if torch.cuda.is_available() else 'cpu'):
    state = load(weights)
    model = SegmentationModel.create_model(state['model'])
    model.load_state_dict(state['model_state_dict'])
    return predict_images(model, images, device)


def get_inference_model(weights, type = "segmentation", watershed_od = ""):
    return ModelInference(weights, type=type, watershed_od=watershed_od)





