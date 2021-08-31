
def train(cfg_path):
    from .utils.utils import parse_yaml
    from .trainer.segmentation.segmentation_trainer import SegmentationTrainer
    from .trainer.classification.classification_trainer import ClassificationTrainer
    from .models.segmentation_model import SegmentationModel
    
    
    type = parse_yaml(cfg_path)['type']
    if type == "classification":
        t = ClassificationTrainer(cfg_path=cfg_path)
    elif type == "segmentation":
        t = SegmentationTrainer(cfg_path=cfg_path)
    t.train()
    print(t.test())


def predict(images, weights, device = 'cuda:0' if torch.cuda.is_available() else 'cpu'):
    import torch
    from torch import load
    from .utils.model_utils import predict_images
    from .models.segmentation_model import SegmentationModel
    state = load(weights)
    model = SegmentationModel.create_model(state['model'])
    model.load_state_dict(state['model_state_dict'])
    return predict_images(model, images, device)


def get_inference_model(weights, type = "segmentation"):
    from .inference.inference import ModelInference
    return ModelInference(weights, type=type)





