from .utils.utils import parse_yaml
from .trainer.classification.classification_trainer import ClassificationTrainer
from .trainer.detection.detection_trainer import DetectionTrainer
import torch
from torch import load
from .inference.inference import ModelInference
import sys

def train(cfg_path):
    type = parse_yaml(cfg_path)['type']
    if type == "classification":
        t = ClassificationTrainer(cfg_path=cfg_path)
    elif type == "segmentation":
        from .trainer.segmentation.segmentation_trainer import SegmentationTrainer
        t = SegmentationTrainer(cfg_path=cfg_path)
    elif type == "od":
        t = DetectionTrainer(cfg_path=cfg_path)
    t.train()
    #print("DONE TRAINING")
    #print(t.test())





def get_inference_model(weights, type = "segmentation", watershed_od = ""):
    return ModelInference(weights, type=type, watershed_od=watershed_od)





