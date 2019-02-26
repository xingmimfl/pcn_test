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
    image_name = "images/25.jpg"
    #image_name = "images/20.jpg"
    pnet = pcn.Pnet()
    #pnet.load_state_dict(torch.load("pnet/pnet_190214_iter_1449000_.pth"))
    pnet.load_state_dict(torch.load("pnet/pnet_190219_iter_1999000_.pth"))
    pnet.eval()
    pnet = pnet.cuda(device_id)

    print("---------finishing loading models---------")
    #----laod images---
    img = cv2.imread(image_name) 
    #imgPad = tools.PadImg(img)
    imgPad = img.copy()
    cv2.imwrite("imgPad.jpg", imgPad)

    img180 = cv2.flip(imgPad, 0)
    img90 = cv2.transpose(imgPad)
    imgNeg90 = cv2.flip(img90, 0)

    cv2.imwrite("img180.jpg", img180)
    cv2.imwrite("img90.jpg", img90)
    cv2.imwrite("imgNeg90.jpg", imgNeg90)

    rectangles = tools.detect_face_pnet(img, imgPad, pnet, 0.5, device_id=6)    
    rectangles = tools.NMS(rectangles, 0.5, 'iou')
    rectangles = rectangles.numpy()
    
    count = 0
    imgpad_copy = imgPad.copy()
    for a_rectangle in rectangles:
        tmp_image = imgPad.copy()
        x1, y1, x2, y2, cls_score, rotate_score = a_rectangle
        print("cls_score:\t", cls_score)
        print("rotate_score:\t", rotate_score)
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) 
        cv2.rectangle(tmp_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(imgpad_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(tmp_image,str(cls_score),(x1, y1+5), cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
        cv2.putText(tmp_image,str(rotate_score),(x1, y1+30), cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
        cv2.imwrite(str(count) + ".jpg", tmp_image) 
        count += 1
        #if count >= 20: break

    cv2.imwrite("pnet.jpg", imgpad_copy) 
