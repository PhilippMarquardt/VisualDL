from torch.utils.data import Dataset
import os
import itertools
from skimage import io
import torch
import albumentations as A

class ClassificationDataset(Dataset):
    def __init__(self, root_folder, transform):
        super().__init__()
        self.root_folder = root_folder
        self.transform = transform
        self.num_classes = len(os.listdir(root_folder))
        self.classes = os.listdir(root_folder)
        self.image_class_tuple = list(itertools.chain.from_iterable([[(os.path.join(path, file), self.classes.index(os.path.split(path)[-1])) for file in files] for path, directories, files in os.walk(root_folder)]))
        
    def __len__(self):
        return len(self.image_class_tuple)
        
    def __getitem__(self, idx):
        img_path, label = self.image_class_tuple[idx]
        img = io.imread(img_path)
        img = img/255.
        transform = A.Compose([
            A.Resize(width=256, height=256)
        ])
        if transform:
            img = transform(image = img)['image']
            
        return torch.tensor(img, dtype = torch.float).permute(2, 0, 1), torch.tensor(label, dtype = torch.long)

        