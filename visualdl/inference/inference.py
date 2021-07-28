import segmentation_models_pytorch as smp
from torch import load
from ..utils.model_utils import predict_images
import torch

class ModelInference():
    def __init__(self, weight_path):
        state = load(weight_path, map_location=torch.device('cpu'))
        self.model = eval(state['model'])
        self.model.load_state_dict(state['model_state_dict'])
        self.model.eval()
    def __call__(self, images):
        return self.predict(images)

    def predict(self, images, device = 'cuda:0' if torch.cuda.is_available() else 'cpu'):
        return predict_images(self.model, images, device)