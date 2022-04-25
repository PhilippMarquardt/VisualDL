# VisualDL
## Installation
1. Please install pytorch first 
2. pip install git+https://github.com/PhilippMarquardt/VisualDL
## Information
For training classification/segmentation/object detection and instance segmentation models with a single file structure.

## File structure
The general file structure for training any model is a folder that contains an images and labels folder.

An example file structure looks like this:

    .
    ├── train              # train folder to put into the config file
    │   ├── images          # train images
    │   ├── labels          # train labels              
    ├── valid              # valid folder to put into the config file
    │   ├── images          # validation images
    │   ├── labels          # validation labels  
    
Images should contain png/jpg files and labels should contain pixel values between 0 and n where n is the number of classes-1.

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
* model_names: A list of model names. Must match up with the number of weights
* settings
    *  nc: The number of classes
    *  epochs: The number of epochs
    *  optimizer: The optimizer that is used for training
    *  lr: The learning rate used for training
    *  workers: The number of workers creating the dataset
    *  batch_size: A list of numbers corresponding to each model given in the model names
    *  gradient_accumulation: Gradient accumulation up to this batch size
    *  criterions: A list of loss function. They will be added during training
    *  tensorboard_log_dir: Directory for the tensorboard logs
    *  metrics: A list of different metrics
    *  class_weights: Calculates class weights when set to True
    *  calculate_weight_map: Calculates the weight map if set to True
    *  early_stopping: Early stopping if no improvement up to this number
    *  custom_data: Anything you would want. Gets saved in the model file after training
* transforms:
    * A list of different Transforms from Albumentations    

### Segmentation config
* data
    * train: The path to the training root folder
    * valid: The path to the valid root folder
    * test: The path to the test root folder
    * weights: A list of weights
    * save_folder: The folder where the weights are saved
* models: A list of dictionaries containing a backbone and a decoder
* settings
    * nc: The number of classes 
    * in_channels: The number of channels for the input layer
    * epochs: The number of epochs
    * optimizer: The optimizer that is used for training
    * lr: The learning rate used for training
    *  workers: The number of workers creating the dataset
    *  batch_size: A list of numbers corresponding to each model given in the model names
    *  gradient_accumulation: Gradient accumulation up to this batch size
    *  criterions: A list of loss function. They will be added during training
    *  use_attention: Whether to use attention
    *  class_weights: Calculates class weights when set to True
    *  calculate_weight_map: Calculates the weight map if set to True
    *  calculate_distance_maps: Adds another class to the model that predicts the distance between each foreground object.
    *  early_stopping: Early stopping if no improvement up to this number
    *  custom_data: Anything you would want. Gets saved in the model file after training
* transforms:
    * A list of different Transforms from Albumentations
    





   
