# VisualDL
## Installation
1. Install pytorch first from https://pytorch.org/
2. pip install git+https://github.com/PhilippMarquardt/VisualDL

## Information
For training classification/segmentation/object detection and instance segmentation models with a single file structure.

## File structure
The file structure varies depending on your task:

### For Segmentation and Instance Segmentation
Uses an images/labels folder structure:

    .
    ├── train              # train folder to put into the config file
    │   ├── images          # train images
    │   ├── labels          # train labels              
    ├── valid              # valid folder to put into the config file
    │   ├── images          # validation images
    │   ├── labels          # validation labels  

Images should contain png/jpg files and labels should contain pixel values between 0 and n where n is the number of classes-1.

### For Classification
Uses a root folder with class subfolders:

    .
    ├── train              # train folder to put into the config file
    │   ├── class1          # folder containing all images of class 1
    │   ├── class2          # folder containing all images of class 2
    │   ├── class3          # folder containing all images of class 3              
    ├── valid              # valid folder to put into the config file
    │   ├── class1          # folder containing all images of class 1
    │   ├── class2          # folder containing all images of class 2
    │   ├── class3          # folder containing all images of class 3

Images should be placed directly in their corresponding class folders as png/jpg files.

Models can be trained using a .yaml config file or by providing a dictionary in memory.
Example yaml configs for training a model can be found in visualdl/trainer/detection|instance|segmentation|classification|series

## Training
The training can be started in two lines of python code:

```python
from visualdl import vdl
vdl.train("visualdl/trainer/segmentation/segmentation.yaml")
```

## Inference
The inference api is designed to enable the user to use the models directly in his own code. 

```python
from visualdl import vdl
import cv2
model = vdl.get_inference_model("path_to_your_train_file.pt")
image = cv2.imread("your_image.png")[::-1] #must be provided in rgb
predictions = model.predict([image]) #returns a list with the prediction for each provided image
```

## Config file
### Classification config

* data
    * train: The path to the training root folder
    * valid: The path to the valid root folder
    * test: The path to the test root folder
    * weights: A list of weights
    * save_folder: The folder where the weights are saved
* model_names: A list of model names from torchvision or timm (e.g., "resnext50_32x4d"). All models supported by pytorch-image-models (timm) can be used
* settings
    * nc: The number of classes
    * epochs: The number of epochs
    * optimizer: The optimizer that is used for training (e.g., "AdamW")
    * lr: The learning rate used for training
    * workers: The number of workers creating the dataset
    * batch_size: A list of numbers corresponding to each model given in the model names
    * gradient_accumulation: Gradient accumulation up to this batch size
    * criterions: A list of loss functions (PyTorch losses + "DiceLoss")
    * tensorboard_log_dir: Directory for the tensorboard logs
    * metrics: A list of metrics from torchmetrics with optional parameters
        * format: `{"name": "MetricName", "params": "param1=value1,param2=value2"}`
    * monitor_metric_name: Metric to monitor for early stopping
    * monitor_metric_params: Parameters for the monitored metric
    * class_weights: Calculates class weights when set to True
    * calculate_weight_map: Calculates the weight map if set to True
    * early_stopping: Early stopping if no improvement up to this number
    * custom_data: Custom metadata saved with the model
* transforms: List of Albumentations transforms with parameters
    * format: `TransformName: param1 = value1, param2 = value2`

### Segmentation config
* data
    * train: The path to the training root folder
    * valid: The path to the valid root folder
    * test: The path to the test root folder
    * weights: A list of weights
    * save_folder: The folder where the weights are saved
* models: A list of dictionaries containing:
    * backbone: Model backbone from segmentation_models_pytorch or timm (e.g., "tu-resnest50d")
    * decoder: Decoder architecture (e.g., "Unet")
* settings
    * nc: The number of classes 
    * in_channels: The number of channels for the input layer
    * epochs: The number of epochs
    * optimizer: The optimizer that is used for training
    * lr: The learning rate used for training
    * workers: The number of workers creating the dataset
    * batch_size: A list of numbers corresponding to each model
    * max_image_size: Maximum image size (required for transformer models)
    * scales: Scale factors or "None"
    * use_attention: Whether to use attention mechanisms
    * ignore_index: Index to ignore in loss calculation
    * gradient_accumulation: Gradient accumulation up to this batch size
    * criterions: A list of loss functions (PyTorch losses + segmentation_models_pytorch losses)
    * tensorboard_log_dir: Directory for tensorboard logs
    * metrics: A list of metrics from torchmetrics with optional parameters
        * format: `{"name": "MetricName", "params": "param1=value1,param2=value2"}`
    * monitor_metric_name: Metric to monitor for early stopping
    * monitor_metric_params: Parameters for the monitored metric
    * class_weights: Calculates class weights when set to True
    * calculate_weight_map: Calculates the weight map if set to True
    * calculate_distance_maps: Adds another class for predicting object distances
    * add_object_detection_model: Whether to add object detection capabilities
    * early_stopping: Early stopping if no improvement up to this number
    * custom_data: Custom metadata saved with the model
* transforms: List of Albumentations transforms with parameters
    * format: `TransformName: param1 = value1, param2 = value2`

## Implementation Details

Example configs for each training type can be found in the according subfolder here: https://github.com/hs-analysis/VisualDL/tree/main/visualdl/trainer. 
Note that detection trainer is currently not supported as this is done by InferenceDL (https://github.com/hs-analysis/InferenceDL/tree/main) in 1.5.x HSA KIT. The instance segmentation model supported is just the maskrcnn. Vision Transformer and another MaskRCNN implementation in 1.5.x is again from InferenceDL.

### Model Support
- Classification models: Supports all models from https://github.com/huggingface/pytorch-image-models
- Segmentation models: Supports all models from https://github.com/qubvel-org/segmentation_models.pytorch, which includes all models supported by timm
- Metrics: All metrics from https://github.com/Lightning-AI/torchmetrics are supported
- Losses: All PyTorch native losses are supported, plus "DiceLoss" which is often used in combination with CrossEntropyLoss to mitigate class imbalances
- Transforms: All augmentations from https://github.com/albumentations-team/albumentations are supported

Example configs with various configurations can be found in the repository's trainer subdirectories.
