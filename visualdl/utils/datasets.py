from torch.utils.data import Dataset
import os
import itertools
from skimage import io
import torch
import albumentations as A
import numpy as np
from tqdm import tqdm
import logging

class ClassificationDataset(Dataset):
    def __init__(self, root_folder, transform, class_weights = False):
        super().__init__()
        self.root_folder = root_folder
        self.transform = transform
        self.num_classes = len(os.listdir(root_folder))
        self.classes = os.listdir(root_folder)
        if class_weights:
            tmp = [1 - y for y in [x/sum([len(os.listdir(os.path.join(root_folder, folder))) for folder in os.listdir(root_folder)]) for x in [len(os.listdir(os.path.join(root_folder, folder))) for folder in os.listdir(root_folder)]]]
            self.class_weights = [z + (1 - max(tmp)) for z in tmp]
        self.image_class_tuple = list(itertools.chain.from_iterable([[(os.path.join(path, file), self.classes.index(os.path.split(path)[-1])) for file in files] for path, directories, files in os.walk(root_folder)]))
    def __len__(self):
        return len(self.image_class_tuple)
        
    def __getitem__(self, idx):
        img_path, label = self.image_class_tuple[idx]
        img = io.imread(img_path).astype(np.float32)
        transform = self.transform
        if transform:
            img = transform(image = img)['image']
        img = img/255.
        return torch.tensor(img, dtype = torch.float).permute(2, 0, 1), torch.tensor(label, dtype = torch.long)



class SegmentationDataset(Dataset):
    def __init__(self, folder, transform, class_weights = False):
        self.folder = folder
        self.transform = transform
        self.train_images = [os.path.join(os.path.join(folder, "images"), image) for image in os.listdir(os.path.join(self.folder, "images"))]
        self.train_masks = [os.path.join(os.path.join(folder, "labels"), image) for image in os.listdir(os.path.join(self.folder, "labels"))]
        if class_weights:
            vals = {}
            mask_loader = tqdm(self.train_masks)
            mask_loader.set_description("Calculating class weights")
            for image in mask_loader:
                img = io.imread(image, as_gray=True)
                #img[img > 0] = 1
                unique, counts = np.unique(img, return_counts=True)
                su = float(sum(counts))
                
                for val, cnt in zip(unique, counts):
                    if not val in vals:
                        vals[val] = cnt / su
                    else:
                        vals[val] += cnt / su
            final = []
            for val in vals.values():
                final.append(1 - (val / len(self.train_masks)))
            self.class_weights = final
            
            print(f"Calculated class weights:{self.class_weights}")
        assert len(self.train_images) == len(self.train_masks)

    def __len__(self):
        return len(self.train_images)

    def __getitem__(self, idx):
        img = io.imread(self.train_images[idx]).astype(np.float32)
        mask = io.imread(self.train_masks[idx], as_gray = True).astype(np.float32)
        #mask[mask > 0] = 1.0
        img = img/255.
        if self.transform:
            transformed = self.transform(image = img, mask = mask)
            img = transformed["image"]
            mask = transformed["mask"]
        kernel = np.ones((2, 2), np.uint8)
        import cv2
        # Using cv2.erode() method 
        #mask = cv2.erode(mask, kernel) 
        return torch.tensor(img, dtype = torch.float).permute(2, 0, 1), torch.tensor(mask, dtype = torch.long)


class ImageOnlyDataset(Dataset):
    """
    Dataset without any labels, just plain images for inference.

    Args:
        Dataset (): 
    """
    def __init__(self, folder):
        self.folder = folder
        self.images = [os.path.join(folder, x) for x in os.listdir(folder)]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = io.imread(self.images[idx]).astype(np.float32)
        img = img/255.
        return torch.tensor(img, dtype=torch.float)


        