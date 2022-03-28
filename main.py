from visualdl.trainer.classification.classification_trainer import *
from visualdl.trainer.segmentation.segmentation_trainer import *
import logging
from visualdl.vdl import train


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger().name = ""

    #r"E:\source\repos\VisualDL\visualdl\trainer\classification\classification.yaml"
    #r"E:\source\repos\VisualDL\visualdl\trainer\segmentation\segmentation.yaml"
    #main(r"E:\source\repos\VisualDL\visualdl\trainer\classification\classification.yaml")
    #train(r"visualdl\trainer\detection\detection.yaml")
    #train(r"visualdl\trainer\segmentation\segmentation.yaml")
    #train(r"visualdl\trainer\instance\instance.yaml")
    train(r"visualdl\trainer\series\series.yaml")