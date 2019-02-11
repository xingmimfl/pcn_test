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
EPS = 0.001

if __name__=="__main__":
    image_name = "11284670_165553491000_2.jpg"
    #image_name = "13.jpg"
    pnet = pcn.Pnet()
    pnet.load_state_dict(torch.load("pnet/pnet_190209_iter_1495000_.pth"))
    pnet.eval()
    pnet = pnet.cuda(device_id)

    rnet = pcn.Rnet()
    rnet.load_state_dict(torch.load("rnet/rnet_190209_iter_1495000_.pth"))
    rnet.eval()
    rnet = rnet.cuda(device_id)

    onet = pcn.Onet()
    onet.load_state_dict(torch.load("onet/onet_190209_iter_1495000_.pth"))
    onet.eval()
    onet = rnet.cuda(device_id)
    print("---------finishing loading models---------")
    #----laod images---
    img = cv2.imread(image_name) 
    imgPad = tools.PadImg(img)
    cv2.imwrite("imgPad.jpg", imgPad)

    img180 = cv2.flip(imgPad, 0)
    img90 = cv2.transpose(imgPad)
    imgNeg90 = cv2.flip(img90, 0)

    cv2.imwrite("img180.jpg", img180)
    cv2.imwrite("img90.jpg", img90)
    cv2.imwrite("imgNeg90.jpg", imgNeg90)

    rectangles = tools.detect_face_pnet(img, imgPad, pnet, 0.5, device_id=6)    
    #rectangles = rectangles.cpu().numpy()
    #for a_rectangle in rectangles:
    #    x1, y1, x2, y2, cls_score, rotate_score = a_rectangle
    #    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) 
    #    cv2.rectangle(imgPad, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #cv2.imwrite("pnet.jpg", imgPad) 

    num_of_rects = len(rectangles)
    rnet_input = torch.zeros(num_of_rects, 3, 24, 24)
    imgpad_rows, imgpad_cols, _ = imgPad.shape
    #print(rectangles)
    for i in range(len(rectangles)):
        a_rect = rectangles[i]
        x1, y1, x2, y2, conf, score_conf = a_rect
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        if score_conf > 0.5:
            crop_image = imgPad[y1:y2, x1:x2]
        else:
            tmp = y2
            y2 = imgpad_rows - 1 - y1
            y1 = imgpad_rows - 1 - tmp
            crop_image = img180[y1:y2, x1:x2]

        crop_image = cv2.resize(crop_image, (24, 24)) / 255.0
        crop_image = crop_image.transpose((2, 0, 1))
        crop_image = torch.from_numpy(crop_image.copy())
        rnet_input[i,:] = crop_image

    rnet_input = Variable(rnet_input).cuda(device_id)
    fc5_2, fc6_2, bbox_reg = rnet(rnet_input)
    rectangles = tools.filter_face_rnet(fc5_2, fc6_2, bbox_reg, rectangles, imgPad, img180, threshold=0.5)

    num_of_rects = len(rectangles)
    onet_input = torch.zeros(num_of_rects, 2, 48, 48)
    for i in range(num_of_rects):
        a_rect = rectangles[i]
        x1, y1, x2, y2, conf, angle = a_rect
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        if (math.abs(angle) < EPS):
            crop_image = imgPad[y1:y2, x1:x2]
        elif math.abs(angle - 90) < EPS:
            crop_image = img90[y1:y2, x1:x2]
        elif math.abs(angle + 90) < EPS:
            crop_image = imgNeg90(y1:y2, x1:x2)
        else:
            tmp = y2
            y2 = imgpad_rows - 1 - y1
            y1 = imgpad_rows - 1 - tmp
            crop_image = img180[y1:y2, x1:x2]              

        crop_image = cv2.resize(crop_image, (24, 24)) / 255.0
        crop_image = crop_image.transpose((2, 0, 1))
        crop_image = torch.from_numpy(crop_image.copy())         
        onet_input[i, :] = crop_image

    onet_input = Variable(onet_input).cuda(device_id)
    fc6, rotate_reg, bbox_reg = onet(onet_input)
    rectangles = tools.filter_face_onet(fc6, rotate_reg, bbox_reg, rectangles, imgPad, threshold=0.5)
