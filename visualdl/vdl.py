from visualdl.utils.datasets import InstanceSegmentationDataset
from .utils.utils import parse_yaml
from .trainer.classification.classification_trainer import ClassificationTrainer
#from .trainer.detection.detection_trainer import DetectionTrainer
from .trainer.segmentation.segmentation_trainer import SegmentationTrainer
from .trainer.instance.instance_trainer import InstanceTrainer
from .trainer.series.series_trainer import SeriesTrainer
from .trainer.series.video_trainer import VideoTrainer
from .trainer.mlp.mlp_trainer import MLPTrainer
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
    # elif type == "od":
    #     t = DetectionTrainer(cfg_path=cfg_path)
    elif type == "instance":
        t = InstanceTrainer(cfg_path=cfg_path)
    elif type == "series":
        t = SeriesTrainer(cfg_path=cfg_path)
    elif type == "video":
        t = VideoTrainer(cfg_path=cfg_path)
    elif type == "mlp":
        t = MLPTrainer(cfg_path=cfg_path)
    t.train()
    #print("DONE TRAINING")
    #print(t.test())





def get_inference_model(weights, type = "segmentation", watershed_od = ""):
    return ModelInference(weights, type=type, watershed_od=watershed_od)



# def get_trainer(cfg_path):
#     type = parse_yaml(cfg_path)['type']
#     if type == "classification":
#         t = ClassificationTrainer(cfg_path=cfg_path)
#     elif type == "segmentation":
#         t = SegmentationTrainer(cfg_path=cfg_path)
#     elif type == "od":
#         t = DetectionTrainer(cfg_path=cfg_path)

#     return t



#Von vdl.train(cfg) wird dann trainer = vdl.get_trainer(cfg) und auf dem kann man dann trainer.train und trainer.stop aufrufen
