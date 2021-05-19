import timm
from .model_base import ModelBase
from ..utils.utils import *
from ..utils.datasets import *
import albumentations as A

class ClassificationModel(ModelBase):
    def __init__(self):
        pass
        
    def train(self):
        model = timm.create_model('resnet26', pretrained=True)
        classes = 2
        model.fc = torch.nn.Linear(model.fc.in_features, classes)
        criterions = [torch.nn.CrossEntropyLoss()]
        transform = A.Compose([
            A.Resize(128,128),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
        ])
        dataset = ClassificationDataset(r"E:\source\repos\VisualDL\tests\classification", transform = None)
        loader = get_dataloader(dataset, 1, 0 )
        trainer(model, loader, None, 20, criterions)
        
    def test(self):
        pass
        
    def inference(self):
        pass
        
    def visualize(self):
        pass
        
        
        
    