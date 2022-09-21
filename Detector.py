#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 12:59:39 2022

@author: iovision
"""

import cv2 as cv
import json
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.modeling import build_model
import torch
import numpy as np
from PIL import Image
import time





class Detector:

    def __init__(self):

        # set model and test set
        self.model = 'mask_rcnn_R_50_FPN_3x.yaml'

        # obtain detectron2's default config
        self.cfg = get_cfg() 

        # load values from a file
        # self.cfg.merge_from_file("test.yaml")
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/"+self.model)) 

        # set device to cpu
        #self.cfg.MODEL.DEVICE = "cuda"
        self.cfg.MODEL.DEVICE = "cpu"

        # get weights 
        # self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/"+self.model) 
        self.cfg.MODEL.WEIGHTS = "/home/iovision/return_img/model_final.pth"
        #self.cfg.MODEL.WEIGHTS = "/home/appuser/return_img_repo/model_final.pth"

        # set the testing threshold for this model

        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        
        #self.cfg.DATASETS.TEST = ("fold1")

        # build model from weights
        # self.cfg.MODEL.WEIGHTS = self.convert_model_for_inference()
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5
        self.predictor = DefaultPredictor(self.cfg)

    # build model and convert for inference
    def convert_model_for_inference(self):

        # build model
        model = build_model(self.cfg)

        # save as checkpoint
        torch.save(model.state_dict(), 'checkpoint.pth')

        # return path to inference model
        return 'checkpoint.pth'

    # detectron model
    # adapted from detectron2 colab notebook: https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5    
    def inference(self, file):
        start= time.time()
      
        im = cv.imread(file)
        outputs = self.predictor(im)
        # with open(self.curr_dir+'/data.txt', 'w') as fp:
        #     json.dump(outputs['instances'], fp)
        #     # json.dump(cfg.dump(), fp)
        
        # get metadata
        MetadataCatalog.get("mydataset").thing_classes = ['short_sleeved_shirt', 'long_sleeved_shirt', 'long_sleeved_outwear', 'shorts', 'trousers']
        # visualise
        v = Visualizer(im[:, :, ::-1], metadata=MetadataCatalog.get("mydataset"), scale=1.2)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        # get image
        img1 = cv.cvtColor(v.get_image()[:, :, ::-1], cv.COLOR_BGR2RGB)
        img = Image.fromarray(np.uint8(img1))
        end = time.time()
        a= end - start 
        # write to jpg
        # cv.imwrite('img.jpg',v.get_image())

        return img, a