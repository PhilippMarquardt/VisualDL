from ...dependencies.yolov5.train import main, parse_opt, run
from ...utils.utils import *
import os
import sysconfig
import sys
import subprocess
import argparse


class DetectionTrainer():
    def __init__(self, cfg_path:dict):
        self.cfg = parse_yaml(cfg_path)
        assert self.cfg['type'] == "od", "Provided yaml file must be a od config!"
        self.this_dir = os.path.dirname(os.path.abspath(__file__))
        datafile = os.path.join(self.this_dir, r"../../dependencies/yolov5/coco128.yaml")  
        newfile = os.path.join(self.this_dir, r"../../dependencies/yolov5/hsa.yaml")  
        with open(datafile, "r",  encoding = "utf-8") as handle:
            a = yaml.load(handle, Loader=yaml.FullLoader)
        a['train'] = self.cfg['data']['train']
        a['val'] = self.cfg['data']['valid']
        a['names'] = [f"{i}" for i in range(int(self.cfg['settings']['nc']))]
        a['nc'] = int(self.cfg['settings']['nc'])
        with open(newfile, "w",  encoding = "utf-8") as handle:
            a = yaml.dump(a, handle)

        modelfile = os.path.join(self.this_dir, r"../../dependencies/yolov5/models/yolov5m.yaml")  
        with open(modelfile, "r",  encoding = "utf-8") as handle:
            a = yaml.load(handle, Loader=yaml.FullLoader)
        a['nc'] = int(self.cfg['settings']['nc'])
        with open(modelfile, "w",  encoding = "utf-8") as handle:
            a = yaml.dump(a, handle)
        

    def train(self):
        assert self.cfg['settings']['modelsize'] in ["s", "m", "l"]

        datafile = os.path.join(self.this_dir, r"../../dependencies/yolov5/hsa.yaml") 
        modelfile = os.path.join(self.this_dir, f"../../dependencies/yolov5/models/yolov5{self.cfg['settings']['modelsize']}.yaml") 
        hyperfile = os.path.join(self.this_dir, r"../../dependencies/yolov5/hyp.scratch.yaml") 
        #opts = parse_opt()
        # setattr(opts, "imgsz", self.cfg['settings']['imgsize'])
        # setattr(opts, "epochs", self.cfg['settings']['epochs'])
        # setattr(opts, "batch_size", self.cfg['settings']['batch_size'])
        # #setattr(opts, "rect", True)
        # setattr(opts, "batch", self.cfg['settings']['batch_size'])
        # if self.cfg['data']['weights'] != "None":
        #     setattr(opts, "weights", self.cfg['data']['weights'])
        # setattr(opts, "data", datafile)
        # setattr(opts, "cfg", modelfile)
        # setattr(opts, "device", 0)
        # setattr(opts, "workers", 0)
        # setattr(opts, "hyp", hyperfile)
        # setattr(opts, "project", self.cfg['data']['save_folder'])
        # if "device" in self.cfg['settings']:
        #     setattr(opts, "device", "cpu")
        # setattr(opts, "cache", "disk")
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        if "device" in self.cfg['settings']:
            run(imgsz = self.cfg['settings']['imgsize'], epochs = self.cfg['settings']['epochs'], batch_size = self.cfg['settings']['batch_size'], batch=self.cfg['settings']['batch_size'],
            data = datafile, cfg=modelfile,workers=0, hyp=hyperfile, project=self.cfg['data']['save_folder'], device="cpu", weights = self.cfg['data']['weights'])
        else:
            run(imgsz = self.cfg['settings']['imgsize'], epochs = self.cfg['settings']['epochs'], batch_size = self.cfg['settings']['batch_size'], batch=self.cfg['settings']['batch_size'],
            data = datafile, cfg=modelfile,device=device, workers=0, hyp=hyperfile, project=self.cfg['data']['save_folder'], weights = self.cfg['data']['weights'])
        #print(opts)
        #main(opts)
        pass

    def test(self):
        pass

        