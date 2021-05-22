from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch
import cv2
import numpy as np

def visualize(model, layer, image):
        cam = GradCAM(model=model, target_layer=layer, use_cuda=torch.cuda.is_available())
        heatmap = cv2.applyColorMap(np.uint8(255 * cam(input_tensor=image.unsqueeze(0))[0,:]), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + image.permute(1,2,0).numpy()
        cam = cam / np.max(cam)
        return cam