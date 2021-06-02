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
        
        transform = self.transform
        if transform:
            img = transform(image = img)['image']
        img = img/255.
        return torch.tensor(img, dtype = torch.float).permute(2, 0, 1), torch.tensor(label, dtype = torch.long)



class SegmentationDataset(Dataset):
    def __init__(self, folder, transform):
        self.folder = folder
        self.transform = transform
        self.train_images = [os.path.join(os.path.join(folder, "images"), image) for image in os.listdir(os.path.join(self.folder, "images"))]
        self.train_masks = [os.path.join(os.path.join(folder, "labels"), image) for image in os.listdir(os.path.join(self.folder, "labels"))]
        assert len(self.train_images) == len(self.train_masks)

    def __len__(self):
        return len(self.train_images)

    def __getitem__(self, idx):
        img = io.imread(self.train_images[idx])
        mask = io.imread(self.train_masks[idx], as_gray = True)
        img = img/255.
        if self.transform:
            transformed = self.transform(image = img, mask = mask)
            img = transformed["image"]
            mask = transformed["mask"]
        return torch.tensor(img, dtype = torch.float).permute(2, 0, 1), torch.tensor(mask, dtype = torch.long)


        