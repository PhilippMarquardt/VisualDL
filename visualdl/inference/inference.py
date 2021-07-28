import segmentation_models_pytorch as smp
from torch import load
from ..utils.model_utils import predict_images

class ModelInference():
    def __init__(self, weight_path):
        state = load(weight_path)
        self.model = eval(state['model'])
        self.model.load_state_dict(state['model_state_dict'])

    def __call__(self, images):
        return self.predict(images)

    def predict(self, images):
        return predict_images(self.model, images)