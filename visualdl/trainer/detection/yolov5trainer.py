from ...dependencies.yolov5.train import main, parse_opt, run
from ...utils.utils import *
import os
import sysconfig
import sys
import subprocess
import argparse



def create_files(folder, nc = 1, single_class = False):
    '''
    Creates yolo dataset
    '''
    files = {}
    start = os.path.join(folder, "labels")
    all_files = os.listdir(start)
    if single_class:
        nc = 1
    for img in os.listdir(start):
        for i in range(0, nc):
            im = cv2.imread(os.path.join(start, img), 0)
            #kernel = np.ones((2, 2), np.uint8)
            #im = cv.erode(im, kernel)
            #im = cv.dilate(im, kernel)
            tmp = im.copy()
            if not single_class:
                tmp[tmp != (i+1)] = 0
                tmp[tmp == (i+1)] = 255
            else:
                tmp[tmp > 0] = 255
            contours,hierachy = cv2.findContours(tmp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            blank = np.zeros_like(tmp)
            for cnt, cont in enumerate(contours):
                xmin,ymin,width,height = cv2.boundingRect(cont)
                #cv.rectangle(im,(x,y),(x+width,y+height),(255),1)
                #cv2.imwrite("xd.png", im)
                image_width = im.shape[0]
                xcenter, ycenter = xmin + width/2, ymin + height/2
                xcenter, ycenter, width, height = xcenter/image_width, ycenter/image_width, width/image_width, height/image_width
                if not img in files:
                    files[img] = [(str(i),str(xcenter), str(ycenter), str(width), str(height))]
                else:
                    files[img] += [(str(i),str(xcenter), str(ycenter), str(width), str(height))]

    filelist = [ f for f in os.listdir(os.path.join(folder, "labels"))]
    for f in filelist:
        os.remove(os.path.join(os.path.join(folder, "labels"), f))
    no_anno = [item for item in all_files if item not in list(files.keys())]
    for cnt, (key, val) in enumerate(files.items()):
        with open(os.path.join(os.path.join(folder, "labels"), key.replace(".png", ".txt").replace(".jpg", ".txt")), "w") as handle:
            for va in val:
                handle.write(" ".join(list(va))+ "\n") 

    for name in no_anno:
        with open(os.path.join(os.path.join(folder, "labels"), name.replace(".png", ".txt").replace(".jpg", ".txt")), "w") as handle:
            handle.write("")

    return files

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
        #print(a)
        with open(newfile, "w",  encoding = "utf-8") as handle:
            a = yaml.dump(a, handle)
        #print(newfile)
        
        
        modelfile = os.path.join(self.this_dir, f"../../dependencies/yolov5/models/yolov5{self.cfg['settings']['modelsize']}.yaml")  
        with open(modelfile, "r",  encoding = "utf-8") as handle:
            a = yaml.load(handle, Loader=yaml.FullLoader)
        a['nc'] = int(self.cfg['settings']['nc'])
        with open(modelfile, "w",  encoding = "utf-8") as handle:
            a = yaml.dump(a, handle)
        
        #prepare dataset
        if self.cfg['data']['mask_input']:
            create_files(self.cfg['data']['train'], int(self.cfg['settings']['nc']))
            create_files(self.cfg['data']['valid'], int(self.cfg['settings']['nc']))

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
            run(custom_data=self.cfg['settings']['custom_data'], imgsz = self.cfg['settings']['imgsize'], epochs = self.cfg['settings']['epochs'], batch_size = self.cfg['settings']['batch_size'], batch=self.cfg['settings']['batch_size'],
            data = datafile, cfg=modelfile,workers=0, hyp=hyperfile, project=self.cfg['data']['save_folder'], device="cpu", weights = self.cfg['data']['weights'])
        else:
            run(custom_data=self.cfg['settings']['custom_data'], imgsz = self.cfg['settings']['imgsize'], epochs = self.cfg['settings']['epochs'], batch_size = self.cfg['settings']['batch_size'], batch=self.cfg['settings']['batch_size'],
            data = datafile, cfg=modelfile,device=device, workers=0, hyp=hyperfile, project=self.cfg['data']['save_folder'], weights = self.cfg['data']['weights'])
        #print(opts)
        #main(opts)
        pass

    def test(self):
        pass

        