import segmentation_models_pytorch as smp
from torch import load
from ..utils.model_utils import predict_images
import torch

class ModelInference():
    def __init__(self, weight_path, device = 'cuda:0' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        if device.lower() == "cpu":
            state = load(weight_path, map_location=torch.device('cpu'))
        else:
            state = load(weight_path)
        self.state = state
        self.model = eval(state['model'])
        self.model.load_state_dict(state['model_state_dict'])
        self.model.eval()

    def __call__(self, images):
        return self.predict(images)

    def predict(self, images, single_class_per_contour = False, min_size = None):
        return predict_images(self.model, images, self.device, single_class_per_contour, min_size)