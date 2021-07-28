from .utils.utils import parse_yaml
from .trainer.segmentation.segmentation_trainer import SegmentationTrainer
from .trainer.classification.classification_trainer import ClassificationTrainer
from .models.segmentation_model import SegmentationModel
from .utils.model_utils import predict_images
from torch import load
from .inference.inference import ModelInference


def train(cfg_path):
    type = parse_yaml(cfg_path)['type']
    if type == "classification":
        t = ClassificationTrainer(cfg_path=cfg_path)
    elif type == "segmentation":
        t = SegmentationTrainer(cfg_path=cfg_path)
    t.train()
    print(t.test())


def predict(images, weights):
    state = load(weights)
    model = SegmentationModel.create_model(state['model'])
    model.load_state_dict(state['model_state_dict'])
    return predict_images(model, images)


def get_inference_model(weights):
    return ModelInference(weights)





