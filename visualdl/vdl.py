from .utils.utils import parse_yaml
from .trainer.segmentation.segmentation_trainer import SegmentationTrainer
from .trainer.classification.classification_trainer import ClassificationTrainer



def main(cfg_path):
    type = parse_yaml(cfg_path)['type']
    if type == "classification":
        t = ClassificationTrainer(cfg_path=cfg_path)
    elif type == "segmentation":
        t = SegmentationTrainer(cfg_path=cfg_path)
    t.train()
    print(t.test())


