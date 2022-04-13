# VisualDL
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


   
