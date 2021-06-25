from visualdl.trainer.classification.classification_trainer import *
from visualdl.trainer.segmentation.segmentation_trainer import *
import logging
from visualdl.vdl import main


logging.getLogger().setLevel(logging.INFO)
logging.getLogger().name = ""

#r"E:\source\repos\VisualDL\visualdl\trainer\classification\classification.yaml"
#r"E:\source\repos\VisualDL\visualdl\trainer\segmentation\segmentation.yaml"
#main(r"E:\source\repos\VisualDL\visualdl\trainer\classification\classification.yaml")
main(r"E:\source\repos\VisualDL\visualdl\trainer\segmentation\segmentation.yaml")