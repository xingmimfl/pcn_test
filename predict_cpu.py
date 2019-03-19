import os
import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
import pcn
import cv2
import numpy as np
import tools_matrix_cpu as tools

threshold = [0.6, 0.6, 0.7]
EPS = 0.001

if __name__=="__main__":
    #image_name = "images/25.jpg"
    #image_name = "images/yueyu.jpg"
    image_name = "images/qingxie.jpg"
    #image_name = "images/9.jpg"
    #image_name = "images/17.jpg"
    #image_name = "images/24.jpg"
    #image_name = "images/5.jpg"
    #image_name = "images/7.jpg"
    #image_name = "images/daozhi.jpg"
    pnet = pcn.Pnet()
    pnet.load_state_dict(torch.load("pnet/pnet_190310_iter_1238000_.pth", map_location=lambda storage, loc: storage))
    pnet.eval()

    rnet = pcn.Rnet()
    rnet.load_state_dict(torch.load("rnet/pnet_190312_iter_979000_.pth", map_location=lambda storage, loc: storage))
    rnet.eval()

    onet = pcn.Onet()
    onet.load_state_dict(torch.load("onet/onet_190227_iter_1499000_.pth", map_location=lambda storage, loc: storage))
    onet.eval()

    print("---------finishing loading models---------")
    #----laod images---
    img = cv2.imread(image_name) 
    #imgPad = tools.PadImg(img)
    imgPad = img.copy()
    img_p_copy = img.copy()
    img_r_copy = img.copy()
    img_o_copy = img.copy()
    cv2.imwrite("imgPad.jpg", imgPad)

    img180 = cv2.flip(imgPad, 0)
    img90 = cv2.transpose(imgPad)
    imgNeg90 = cv2.flip(img90, 0)

    cv2.imwrite("img180.jpg", img180)
    cv2.imwrite("img90.jpg", img90)
    cv2.imwrite("imgNeg90.jpg", imgNeg90)

    rectangles = tools.detect_face_pnet(img, imgPad, pnet, 0.5, device_id=6)    
    #rectangles = rectangles.cpu().numpy()
    for a_rectangle in rectangles:
        x1, y1, x2, y2, cls_score, rotate_score = a_rectangle
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) 
        cv2.rectangle(img_p_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite("pnet.jpg", img_p_copy) 

    #----rnet-------
    img_height, img_width, _ = img.shape
    num_of_rects = len(rectangles)
    rnet_input = torch.zeros(num_of_rects, 3, 24, 24)
    count = 0
    for i in range(len(rectangles)):
        a_rect = rectangles[i]
        x1, y1, x2, y2, conf, score_conf = a_rect
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        if abs(score_conf) < EPS: #---人脸向上
            crop_image = img[y1:y2, x1:x2]
        else:
            y1, y2 = img_height - 1 - y2, img_height - 1 - y1
            crop_image = img180[y1:y2, x1:x2]
        cv2.imwrite("rnet_input_"+str(count) + "_" + str(score_conf) + ".jpg", crop_image) 
        count += 1
        crop_image = cv2.resize(crop_image, (24, 24)) / 255.0
        crop_image = crop_image.transpose((2, 0, 1))
        crop_image = torch.from_numpy(crop_image.copy())
        rnet_input[i,:] = crop_image

    rnet_input = Variable(rnet_input)
    fc5_2, fc6_2, bbox_reg = rnet(rnet_input)
    rectangles = tools.filter_face_rnet(fc5_2, fc6_2, bbox_reg, rectangles, img, img180, threshold=0.5)
    print(rectangles) 
    for a_rectangle in rectangles:
        x1, y1, x2, y2, cls_score, rotate_score = a_rectangle
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(img_r_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite("rnet.jpg", img_r_copy)

    #----onet predict -    
    height, width, _ = img.shape
    num_of_rects = len(rectangles) 
    print("num_of_rects:\t",num_of_rects)
    onet_input = torch.zeros(num_of_rects, 3, 48, 48)
    count = 0
    for i in range(num_of_rects):
        a_rect = rectangles[i]     
        x1, y1, x2, y2, conf, rotate_angle = a_rect
        print("rotate_angle:\t",rotate_angle)
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        if abs(rotate_angle) < EPS:
            crop_image = img[y1:y2, x1:x2] 
        elif abs(rotate_angle - 90) < EPS:
            crop_image = img90[x1:x2, y1:y2]   
        elif abs(rotate_angle + 90) < EPS:
            crop_image = imgNeg90[width-1-x2:width-1-x1, y1:y2]         
        else:
            crop_image = img180[height-1-y2:height-1-y1, x1:x2]  

        cv2.imwrite("onet_input_"+str(count) + "_" + str(rotate_angle) + ".jpg", crop_image) 
        count += 1
        
        crop_image = cv2.resize(crop_image, (48, 48))/255.0
        crop_image = crop_image.transpose(2, 0, 1)
        crop_image = torch.from_numpy(crop_image.copy())
        onet_input[i,:]  = crop_image 

    onet_input = Variable(onet_input)
    fc6, rotate_reg, bbox_reg = onet(onet_input)
    rectangles = tools.filter_face_onet(fc6, rotate_reg, bbox_reg, rectangles, img, img180, img90, imgNeg90,  0.5) 
    for a_rectangle in rectangles:
        x1, y1, x2, y2, cls_score, rotate_score = a_rectangle
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(img_o_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_o_copy, str(rotate_score), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0))  
    cv2.imwrite("onet.jpg", img_o_copy)    
