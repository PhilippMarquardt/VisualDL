import segmentation_models_pytorch as smp
from torch import load
from ..utils.model_utils import predict_images
import torch


def predict_od(model, imgs, confidence = 0.45):
    size = imgs[0].shape[0]
    model.conf = confidence
    preds = model(imgs, size=size)
    finals = []
    for cnt, img in enumerate(imgs):
        tmp = []
        boxes = preds.xyxy[cnt]
        for box in boxes:
            middlex = int(box[0] + (box[2] - box[0]) / 2)
            middley = int(box[1] + (box[3] - box[1]) / 2)
            data = list(box.detach().cpu().numpy())
            data.append((middlex, middley))
            tmp.append(tuple(data))
        finals.append(tmp)
    return finals

class ModelInference():
    def __init__(self, weight_path, device = 'cuda:0' if torch.cuda.is_available() else 'cpu', type='segmentation'):
        self.device = device
        self.type = type
        if type == "segmentation":
            if device.lower() == "cpu":
                state = load(weight_path, map_location=torch.device('cpu'))
            else:
                state = load(weight_path)
            self.state = state
            self.model = eval(state['model'])
            self.model.load_state_dict(state['model_state_dict'])
            self.model.eval()
        
        elif type == "od":
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=weight_path)

    def __call__(self, images):
        return self.predict(images)

    def predict(self, images, single_class_per_contour = False, min_size = None, confidence = 0.45):
        if self.type == "segmentation":
            return predict_images(self.model, images, self.device, single_class_per_contour, min_size)
        elif self.type == "od":
            return predict_od(self.model, images, confidence=confidence)