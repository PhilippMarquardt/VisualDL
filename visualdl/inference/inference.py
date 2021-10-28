import segmentation_models_pytorch as smp
from torch import load
from ..utils.model_utils import predict_images, make_single_class_per_contour
import torch
import numpy as np
import os
import cv2
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.morphology import square
from ..models.dpt.models import DPTSegmentationModel
from ..models.dpt.models import DPTSegmentationModel
from ..models.custom import U2NET
from ..models.TransInUnet import TransInUnet
from ..models.hrnet import HRNetV2
from ..models.caranet.CaraNet import caranet
from ..models.doubleunet.doubleunet import DoubleUnet
from ..models.unetr import UNETR

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
    def __init__(self, weight_path, device = 'cuda:0' if torch.cuda.is_available() else 'cpu', type='segmentation', watershed_od = ""):
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
            self.has_distance_map = False
            if 'has_distance_map' in self.state:
                self.has_distance_map = bool(self.state['has_distance_map'])
        
        elif type == "od":
            dirname = os.path.dirname(__file__)
            dirname = dirname.replace(dirname.split("\\")[-1], "dependencies/yolov5")
            self.state = {}
            self.model, self.state['custom_data'] = torch.hub.load(dirname, 'custom', path=weight_path, source='local', return_custom_data = True)
            

        elif type == "segmentation_od":
            if device.lower() == "cpu":
                state = load(weight_path, map_location=torch.device('cpu'))
            else:
                state = load(weight_path)
            self.state = state
            self.model = eval(state['model'])
            self.model.load_state_dict(state['model_state_dict'])
            self.model.eval()
            self.has_distance_map = False
            if 'has_distance_map' in self.state:
                self.has_distance_map = bool(self.state['has_distance_map'])

            self.model_od = torch.hub.load('ultralytics/yolov5', 'custom', path=watershed_od)

    def __call__(self, images):
        return self.predict(images)

    def predict(self, images, single_class_per_contour = False, min_size = None, confidence = 0.45, fill_holes = True):
        if self.type == "segmentation":
            return predict_images(self.model, images, self.device, single_class_per_contour, min_size, self.has_distance_map, fill_holes=fill_holes)
        elif self.type == "od":
            return predict_od(self.model, images, confidence=confidence)
        elif self.type == "segmentation_od":
            all_segmentations = []
            boxes =  predict_od(self.model_od, images, confidence=confidence)
            maps = predict_images(self.model, images, self.device, single_class_per_contour, min_size, self.has_distance_map, fill_holes = fill_holes)[0]
            
            #images must be rgb
            for image, box, map in zip(images, boxes, maps):
                label_map = np.int32(np.zeros_like(map))
                p = 1
                for b in box:
                    label_map = cv2.circle(label_map, b[-1], 1, p, -1)
                    p += 1





                #ndi waterhsed    
                distance = ndi.distance_transform_edt(map)
                #mapss = np.uint8(ndi.binary_fill_holes(map))
                labels = watershed(-distance, label_map, mask=map, watershed_line = True)
                labels[labels > 0] = map[labels>0]
                kernel = np.ones((2, 2), np.uint8)
                labels = cv2.erode(np.uint8(labels), kernel)


                # rgb_mask = cv2.cvtColor(map.astpye(np.uint8), cv2.COLOR_GRAY2RGB)
                # markers = cv2.watershed(rgb_mask, label_map)
                # empty = np.zeros_like(markers).astype(np.uint8)
                # empty[markers == -1] = 255
                # kernel = np.ones((2, 2), np.uint8)
                # labels = cv2.dilate(empty, kernel)
                # maps[labels == 255] = 0
                all_segmentations.append(make_single_class_per_contour(labels, min_size))
            return all_segmentations
            
            
            
