# VisualDL
## Installation
1. Install pytorch first from https://pytorch.org/
2. pip install git+https://github.com/hs-analysis/VisualDL

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
The inference API is designed to enable users to use the models directly in their own code. The `get_inference_model` function returns a `ModelInference` instance configured for the specific model type.

```python
from visualdl import vdl
import cv2

# Basic usage
model = vdl.get_inference_model("path_to_your_train_file.pt")  # defaults to segmentation type
image = cv2.imread("your_image.png")[::-1]  # must be provided in rgb
predictions = model.predict([image])  # returns a list with the prediction for each provided image

# Specifying model type
model = vdl.get_inference_model(
    weights="path_to_your_train_file.pt",
    type="classification"  # options: "segmentation", "classification", "od", "instance", "series"
)

# Note: The watershed_od parameter is technically supported but deprecated
# and should not be used in new applications
model = vdl.get_inference_model(
    weights="path_to_your_train_file.pt",
    watershed_od=""  # deprecated parameter
)
```

Implementation details:
```python
def get_inference_model(weights, type="segmentation", watershed_od=""):
    return ModelInference(weights, type=type, watershed_od=watershed_od)

def predict(
    self,
    images,
    single_class_per_contour=False,
    min_size=None,
    confidence=0.45,
    fill_holes=False,
    mlp_output_type="default",
):
    """
    Predict function for inference.
    
    Args:
        images: List of input images in RGB format
        single_class_per_contour: For segmentation models - forces each detected contour to have a single class.
                                 This can improve results in cases where the model predicts multiple classes
                                 within what should be a single object.
        min_size: Minimum size threshold for detected regions
        confidence: Confidence threshold (particularly important for instance segmentation models,
                   default is 0.45)
        fill_holes: Whether to fill holes in the predictions
        mlp_output_type: Type of MLP output processing, defaults to "default"
    
    Returns:
        List of predictions corresponding to input images
    """
```
### Segmentation Inference Example for folder
```python
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from visualdl import vdl
import cv2
import numpy as np
pred = vdl.get_inference_model(r"C:\Users\philmarq\source\repos\VisualDL\megfinalglyphe\efficientnet-b7, UnetPlusPlus.pt")

folder = r"C:\Users\philmarq\source\repos\Daten\dataset\dataset\valid\images"
image_size = 512#
output_folder = "outputall/" 
for file in os.listdir(folder):
    img = cv2.cvtColor(
    cv2.imread(
        os.path.join(folder, file)
    ),
    cv2.COLOR_BGR2RGB,
    )
    orig_shape = img.shape
    img = cv2.resize(img, (image_size, image_size))
    
    preds = pred.predict([img], single_class_per_contour = False)
    preds = preds[0][0] * 50 #*50 just for visualization
    preds = cv2.resize(preds.astype(np.float32), (orig_shape[1], orig_shape[0]))
    cv2.imwrite(os.path.join(output_folder, file), preds)
```
### Instance Segmentation Example
Here's a complete example showing how to use instance segmentation model inference, including visualization:

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import random
from visualdl import vdl

def run_instance_segmentation():
    # Load the model
    model = vdl.get_inference_model("path_to_model.pt", type="instance")
    
    # Read and preprocess image
    image = cv2.imread("path_to_image.jpg")
    if image is None:
        raise ValueError("Could not load image")
    
    # Convert BGR to RGB and resize
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
    
    # Run inference
    start_time = time.time()
    predictions = model.predict([image], confidence=0.35)[0]
    inference_time = time.time() - start_time
    print(f"Inference time: {inference_time:.3f} seconds")
    
    # Process predictions
    # The model returns a tuple of (boxes, classes, scores, masks)
    boxes = predictions[0]  # [N, 4] array of bounding boxes
    masks = predictions[3]  # [N, 1, H, W] array of segmentation masks
    scores = predictions[2]  # [N] array of confidence scores
    
    # Create visualization
    final_mask = np.zeros((image.shape[0], image.shape[1]))
    threshold = 0.25  # Mask threshold
    
    # Process each detection
    for box, mask, score in zip(boxes, masks, scores):
        if score.item() > 0.35:  # Confidence threshold
            # Threshold the mask
            binary_mask = mask[0].copy()  
            binary_mask[binary_mask < threshold] = 0
            binary_mask[binary_mask >= threshold] = 255
            
            # Only add mask if it doesn't overlap too much
            if np.count_nonzero(final_mask[binary_mask == 255]) <= 5000:
                final_mask[binary_mask == 255] = random.randint(50, 255)
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original image
    ax1.imshow(image)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Segmentation mask
    ax2.imshow(final_mask, cmap='tab20')
    ax2.set_title('Instance Segmentation')
    ax2.axis('off')
    
    plt.tight_layout()
    return fig

# Run the example
fig = run_instance_segmentation()
plt.show()
```

Key points about instance segmentation inference:
- The model's output is a tuple of (boxes, classes, scores, masks)
- Use appropriate confidence thresholds (e.g., 0.35) to filter predictions
- Mask threshold (0.25) controls the binary segmentation boundary
- Consider overlap checking to prevent instances from merging
- The example includes visualization of both input image and predictions

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
