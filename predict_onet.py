import os
import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
import pcn
import cv2
import numpy as np
import tools_matrix as tools

device_id = 6
threshold = [0.6, 0.6, 0.7]

if __name__=="__main__":
    #image_name = "images/yueyu.jpg"
    image_name = "onet_input_0_-90.0.jpg"
    #image_name = "images/20.jpg"
    onet = pcn.Onet()
    #onet.load_state_dict(torch.load("onet/onet_190214_iter_1449000_.pth"))
    onet.load_state_dict(torch.load("onet/onet_190227_iter_1499000_.pth", map_location=lambda storage, loc: storage))
    onet.eval()

    print("---------finishing loading models---------")
    #----laod images---
    img = cv2.imread(image_name) 
    img = cv2.resize(img, (48, 48)) / 255.0
    img = img.transpose((2,0,1))
    img = torch.from_numpy(img.copy())
    img = torch.unsqueeze(img, 0)
    img = Variable(img).float()
    print("----processing------")
    fc5, rotate_reg, bbox_reg = onet(img)
    print(rotate_reg * 45)
    print(bbox_reg)
