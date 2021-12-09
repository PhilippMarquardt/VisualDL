from torch.utils.data import Dataset
import os
import itertools
from skimage import io
import torch
import albumentations as A
import numpy as np
from tqdm import tqdm
import logging
import cv2

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

class InstanceSegmentationDataset(Dataset):
    def __init__(self, folder, transform = None, use_cache = True):
        self.folder = folder
        self.train_images = [os.path.join(os.path.join(folder, "images"), image) for image in os.listdir(os.path.join(self.folder, "images"))]
        self.train_masks = [os.path.join(os.path.join(folder, "labels"), image) for image in os.listdir(os.path.join(self.folder, "labels"))]
        self.transform = transform
        self.use_cache = use_cache
        self.cached_data = []
        to_delete = []
        self.max_boxes = 0
        for im, la in zip(self.train_images, self.train_masks):
           mask = cv2.imread(la, 0) #Currently changed to the original image   
           if np.count_nonzero(mask) == 0:
               to_delete.append((im, la))
        for im, la in to_delete:
           self.train_images.remove(im)
           self.train_masks.remove(la)
        
    def __len__(self):
        return len(self.train_images)

    # def __getitem__(self, idx):
    #      while idx < len(self.train_images):
    #         sample = self._get_item_safe(idx)
    #         if sample is not None:
    #             return sample
    #         idx+=1
    def __getitem__(self, idx):
        img = io.imread(self.train_images[idx]).astype(np.float32)
        img = img / 255.
        mask = io.imread(self.train_masks[idx], as_gray = True)#.astype(np.float32)

        if self.transform:
            transformed = self.transform(image = img, mask = mask)
            img = transformed["image"]
            mask = transformed["mask"]


        tmp = mask.copy()
        tmp[tmp > 0] = 1
        contours,hierachy = cv2.findContours(tmp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        labels = []
        masks = []
        
        for cnt, cont in enumerate(contours):
            rect = cv2.boundingRect(cont)
            xmin, ymin, xmax, ymax = rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]
            rect = (xmin, ymin, xmax, ymax)
            mask2 = np.zeros_like(tmp)
            cv2.drawContours(mask2, [cont], -1, 255, -1)
            pts = np.where(mask2 > 0)
            _cls = mask[pts[0], pts[1]]
            cls = np.unique(_cls)[np.argmax(np.unique(_cls, return_counts = True)[1])]
            mask2[pts[0], pts[1]] = mask[pts[0], pts[1]]
            boxes.append(rect)
            labels.append(cls.astype(np.int64))
            mask2[mask2>0] = 1 #somehow needs to be binary
            masks.append(mask2.astype(np.uint8))
        if len(contours) != 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
        else:
             boxes = torch.empty((0, 4), dtype=torch.float32)
        if len(contours) != 0:
            labels = torch.as_tensor(labels, dtype=torch.int64)
        else:
            labels = torch.as_tensor([0], dtype=torch.int64)
        if len(contours) != 0:
            masks = torch.as_tensor(masks, dtype=torch.uint8)
        else:
            masks = torch.as_tensor(torch.empty(0, 512, 512), dtype=torch.uint8)
        if len(contours) != 0:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        else:
            area = torch.tensor([0], dtype=torch.float32)
        is_crowd = torch.zeros((len(contours),), dtype=torch.int64)
        image_id = torch.tensor([idx])
        target = dict()
        self.max_boxes = max(self.max_boxes, len(contours))
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = is_crowd
        #self.cached_data.append() TODO: probably add to cached data but poor RAM :( 
        return torch.tensor(img, dtype=torch.float).permute(2,0,1), target

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


        