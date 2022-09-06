import timm
from timm.models.factory import safe_model_name
from torch.utils.data.dataloader import DataLoader
from .model_base import ModelBase
from ..utils.utils import get_dataloader, write_image
import albumentations as A
from torch.nn import *
import logging
import torch
from skimage import io
import numpy as np
import cv2
from ..utils.model_utils import evaluate, visualize, train_all_epochs
from torch.utils.tensorboard import SummaryWriter
from torch.optim import *


class ClassificationModel(ModelBase):
    def __init__(
        self,
        name,
        nc,
        criterions,
        metrics,
        monitor_metric,
        optimizer,
        lr,
        accumulate_batch,
        tensorboard_dir,
        class_weights,
        weight,
        save_folder,
        early_stopping,
        custom_data,
    ):
        super().__init__(
            nc,
            criterions,
            metrics,
            monitor_metric,
            optimizer,
            lr,
            accumulate_batch,
            tensorboard_dir,
            class_weights,
            model=timm.create_model(name, pretrained=True, num_classes=nc),
            calculate_weight_map=False,
            weight=weight,
            save_folder=save_folder,
            early_stopping=early_stopping,
            calculate_distance_maps=False,
            custom_data=custom_data,
        )
        self.name = name
        self.modelstring = (
            f"timm.create_model('{name}', pretrained={False}, num_classes = {nc})"
        )

    def __call__(self, x):
        return self.model(x)

    def train(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader = None,
        test_loader: DataLoader = None,
        epochs: int = 1,
    ):
        """Trains the model on the given DataLoader.

        Args:
            train_loader (DataLoader): The training DataLoader
            valid_loader (DataLoader, optional): The valid DataLoader. Defaults to None.
            test_loader (DataLoader, optional): The test DataLoader. Defaults to None.
            epochs (int, optional): The number of epochs. Defaults to 1.
        """
        if hasattr(train_loader.dataset, "class_weights"):
            for crit in self.criterions:
                if crit.weight is None:
                    crit.weight = torch.tensor(train_loader.dataset.class_weights).to(
                        "cuda:0" if torch.cuda.is_available() else "cpu"
                    )
        train_all_epochs(
            model=self.model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            test_loader=test_loader,
            epochs=epochs,
            criterions=self.loss,
            metrics=self.metrics,
            monitor_metric=self.monitor_metric,
            writer=self.writer,
            name=self.name,
            optimizer=self.optimizer,
            accumulate_batch=self.accumulate_batch,
            weight_map=self.calculate_weight_map,
            save_folder=self.save_folder,
            early_stopping=self.early_stopping,
            modelstring=self.modelstring,
            custom_data=self.custom_data,
        )

    def test(self, test_loader):
        pass

    def visualize(self, image):
        return visualize(self.model, self.model.layer4[-1], image)
