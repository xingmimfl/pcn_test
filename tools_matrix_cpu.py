# -*- coding: UTF-8 -*-
import sys
from operator import itemgetter
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

'''
Function:
    change rectangles into squares (matrix version)
Input:
    rectangles:  torch.Tensor
        rectangles[i][0:3] is the position, rectangles[i][4] is score
Output:
    squares: same as input
'''
def rect2square(rectangles):
    w = rectangles[:,2] - rectangles[:,0]
    h = rectangles[:,3] - rectangles[:,1]
    l, max_index = torch.max(torch.stack([w, h], dim=1), dim=1)
    rectangles[:,0] = rectangles[:,0] + w*0.5 - l*0.5
    rectangles[:,1] = rectangles[:,1] + h*0.5 - l*0.5
    rectangles[:,2:4] = rectangles[:,0:2] + l.view(-1, 1).repeat(1, 2)
    return rectangles

def NMS(rectangles, threshold, type):
    """
    Function:
        apply NMS(non-maximum suppression) on ROIs in same scale(matrix version)
    Input:
        rectangles: torch.floattensor. rectangles[i][0:3] is the position, rectangles[i][4] is score
    Output:
        rectangles: same as input
    """
    if rectangles.size(0)==0:
        return rectangles

    x1 = rectangles[:, 0]
    y1 = rectangles[:, 1]
    x2 = rectangles[:, 2]
    y2 = rectangles[:, 3]
    score = rectangles[:, 4]
    area = torch.mul(x2 - x1 + 1.0, y2 - y1 + 1.0)  
    _, order = torch.sort(score, descending=True)
    keep = []
    while order.numel() > 0:
        idx = order[0]  #---highest score
        keep.append(idx)
        if order.size(0) ==1: break
        xx1 = torch.clamp(x1[order[1:]], min=x1[idx])
        yy1 = torch.clamp(y1[order[1:]], min=y1[idx])
        xx2 = torch.clamp(x2[order[1:]], max=x2[idx])
        yy2 = torch.clamp(y2[order[1:]], max=y2[idx])
        
        w = torch.clamp(xx2 - xx1 + 1, min=0.0)
        h = torch.clamp(yy2 - yy1 + 1, min=0.0)
        inter  = torch.mul(w, h)
        if type == 'iom':
            o = inter / torch.clamp(area[order[1:]], max=area[idx])
        else:
            o = inter / (area[idx] + area[order[1:]] - inter)
        order = order[1:][(o < threshold)]

    keep = torch.LongTensor(keep) #---convert list to torch.LongTensor
    result_rectangles = torch.index_select(rectangles, 0, keep) 
    return result_rectangles


         
def detect_face_pnet(img, imgPad, net, thres, device_id=0):
    """
    img: original image
    imgPad: padded original image
    net: net model
    thres: threshold for net 
    """
    img_pad_rows, img_pad_cols, _ = imgPad.shape
    img_rows, img_cols, _ = img.shape
    delta_row = (img_pad_rows - img_rows) / 2
    delta_col = (img_pad_cols - img_cols) /2 

    netSize = 24 #-----size of input
    minFace = 28
    stride = 8
    scale_ = 1.414
    curScale = minFace * 1.0 / netSize  
    imgResized = cv2.resize(img, (int(img.shape[1]/curScale), int(img.shape[0]/curScale)), interpolation=cv2.INTER_CUBIC)

    result = []
    count = 0
    while min(imgResized.shape[:2]) >= netSize:
        #print("curScale:\t", curScale)
        imgResized_input = imgResized.copy()
        imgResized_input = imgResized_input.transpose((2, 0, 1)) / 255.0 #---[H, W, 3] ----> [3, H, W]
        imgResized_input = torch.from_numpy(imgResized_input)
        imgResized_input = torch.unsqueeze(imgResized_input, 0) #----[1, 3, H, W]
        imgResized_input = Variable(imgResized_input).float()
        prob, rotate_prob, bbox_reg = net(imgResized_input) 
        #prob = F.sigmoid(prob)
        #rotate_prob = F.sigmoid(rotate_prob)
        #prob: [1, 1, h, w]; rotate_prob: [1, 1, h, w]; bbox_reg: [1, 4, h, w]
        prob = prob[0, 0]; rotate_prob = rotate_prob[0, 0];  bbox_reg = bbox_reg[0]
        prob = prob.t(); rotate_prob = rotate_prob.t(); bbox_reg = bbox_reg.permute(0, 2, 1); #--[4, w, h]
        prob_mask = (prob > thres)
        indexes = prob_mask.nonzero().float()
        if indexes.numel() > 0:
            bb1 = torch.round((stride * indexes + 0) * curScale) #--[x1, y1]
            bb2 = torch.round((stride * indexes + 16) * curScale) #--[x2, y2]
            boundingbox = torch.cat([bb1, bb2], dim=1)
            dx1, dy1, dx2, dy2 = bbox_reg[0][prob_mask], bbox_reg[1][prob_mask], bbox_reg[2][prob_mask], bbox_reg[3][prob_mask]
            offset = torch.stack([dx1, dy1, dx2, dy2], dim=1)

            boundingbox = boundingbox + offset * 16.0 * curScale
            #boundingbox[:, 0::2] = boundingbox[:, 0::2] + delta_col
            #boundingbox[:, 1::2] = boundingbox[:, 1::2] + delta_row
            score = prob[prob_mask].unsqueeze(1)
            rotate_score = rotate_prob[prob_mask]
            rotate_positive_mask = (rotate_score >= 0.5)
            rotate_negative_mask = ~rotate_positive_mask
            rotate_score[rotate_positive_mask] = 0. #----0 degree
            rotate_score[rotate_negative_mask]  = 180. #----180 degree
            rotate_score = rotate_score.unsqueeze(1) #---change shape
            rectangles = torch.cat([boundingbox, score, rotate_score], dim=1)
            rectangles = rect2square(rectangles) #---turn into square

            rectangles[:, :2] = torch.clamp(rectangles[:, :2], min=0)
            rectangles[:, 2] = torch.clamp(rectangles[:, 2], max=img_cols)
            rectangles[:, 3] = torch.clamp(rectangles[:, 3], max=img_rows)

            index1 = (rectangles[:, 2] > rectangles[:, 0] + 12) #---x2 > x1
            index2 = (rectangles[:, 3] > rectangles[:, 1] + 12) #---y2 > y1
            index = (index1 & index2).nonzero().squeeze()
            rectangles = rectangles.index_select(0, index)

            if rectangles.numel() > 0:
                rectangles = rectangles.data
                rectangles = NMS(rectangles, 0.5, 'iou')
                result.append(rectangles)

        #cv2.imwrite("resized_"+str(count) + ".jpg", imgResized)
        count += 1
        imgResized = cv2.resize(imgResized, (int(imgResized.shape[1]/scale_), int(imgResized.shape[0]/scale_)), interpolation=cv2.INTER_CUBIC)    
        curScale = img_rows * 1.0 / imgResized.shape[0]

    result = torch.cat(result, dim=0)
    return result

def filter_face_rnet(prob, angle_prob, bbox_reg, rectangles, img, img180, threshold):
    """
    prob.size: [N, 1]
    angle_prob.size: [N, 3]
    bbox_reg.size: [N, 4]
    """
    height, width, _ = img.shape
    prob = F.sigmoid(prob) #---sigmoid classification prob
    angle_prob = F.softmax(angle_prob) #---softmax for angle prob
    prob = prob.data; #---variable to tensor
    angle_prob = angle_prob.data; #---variable to tensor
    bbox_reg = bbox_reg.data #---variable to tensor

    #---if face up: index = 0, angle = 90
    #               index = 1, angle = 0
    #               index = 2, angle = -90
    #---if face down:index = 0, angle = 90
    #               index = 1, angle = 180
    #               index = 2, angle = -90
    _, angle_max_index = torch.max(angle_prob, dim=1) #----find corresponding angle
    binary_tensor = (prob>=threshold)
    indexes = binary_tensor.nonzero()[:,0]
    if indexes.numel() < 0: return []
    angle_max_index = torch.index_select(angle_max_index, 0, indexes) #---find positive samples
    rectangles = torch.index_select(rectangles, 0, indexes) #----select rectangles of positive samples
    rectangles[:, 4] = torch.index_select(prob, 0, indexes) #----replace classification score

    face_down_mask = torch.gt(rectangles[:, 5], 10.0) #----mask for face down
    face_down_index = face_down_mask.nonzero() 
     
    face_up_mask = ~face_down_mask #----mask for face up
    face_up_index = face_up_mask.nonzero()

    if face_up_index.numel() > 0:
        face_up_index = face_up_index[:, 0] 
        rectangles_up = torch.index_select(rectangles, 0, face_up_index)  
        angle_max_index_up = torch.index_select(angle_max_index, 0, face_up_index)
        rectangles_up[:, 5] = (angle_max_index_up - 1) * -90
        rectangles[face_up_index] = rectangles_up
        
    if face_down_index.numel() > 0:
        face_down_index = face_down_index[:, 0]
        rectangles_down = torch.index_select(rectangles, 0, face_down_index)      
        angle_max_index_down = torch.index_select(angle_max_index, 0, face_down_index) 
        rectangles_down[:, 5] = (angle_max_index_down - 1) * -90
        angle_max_index_down_1 = torch.eq(angle_max_index_down, 1) #----rotate class label 等于1
        angle_max_index_down_1 = angle_max_index_down_1.nonzero()
        if angle_max_index_down_1.numel() > 0:
            angle_max_index_down_1 = angle_max_index_down_1[:, 0] 
            rectangles_down_tmp = torch.index_select(rectangles_down, 0, angle_max_index_down_1) 
            rectangles_down_tmp[:, 5] = 180.0
            rectangles_down[angle_max_index_down_1] = rectangles_down_tmp            

        rectangles_down[:, 1], rectangles_down[:, 3] = height - 1 - rectangles_down[:, 3], height-1 - rectangles_down[:, 1] #---img180中的坐标
        rectangles[face_down_index] = rectangles_down
     
    #-----deal [x1, y1, x2, y2]------
    bbox_reg = torch.index_select(bbox_reg, 0, indexes)
    w = (rectangles[:, 2] - rectangles[:, 0]).view(-1, 1) #---[N]--->[N, 1]
    h = (rectangles[:, 3] - rectangles[:, 1]).view(-1, 1)
    bbox_reg[:, 0::2] = torch.mul(bbox_reg[:, 0::2], w)
    bbox_reg[:, 1::2] = torch.mul(bbox_reg[:, 1::2], h)
    rectangles[:, :4] = rectangles[:, :4] + bbox_reg
    rectangles = rect2square(rectangles)
    
    #-----坐标换算到原来的图片
    if face_down_index.numel() > 0:
        rectangles_down = torch.index_select(rectangles, 0, face_down_index)
        rectangles_down[:, 1], rectangles_down[:, 3] = height-1 - rectangles_down[:, 3], height-1 - rectangles_down[:, 1] 
        rectangles[face_down_index] = rectangles_down         

    #----legal judgement-----
    rectangles[:, :2] = torch.clamp(rectangles[:, :2], min=0)
    rectangles[:, 2] = torch.clamp(rectangles[:, 2], max=width)
    rectangles[:, 3] = torch.clamp(rectangles[:, 3], max=height)

    index1 = (rectangles[:, 2] > rectangles[:, 0]) #---x2 > x1
    index2 = (rectangles[:, 3] > rectangles[:, 1]) #---y2 > y1
    index = (index1 & index2).nonzero().squeeze()
    rectangles = rectangles.index_select(0, index) 
    rectangles = NMS(rectangles, 0.5, 'iou')
    return rectangles

def filter_face_onet(cls_prob, rotate_reg, bbox_reg, rectangles, img, img180, img90, imgneg90, threshold):
    """
    cls_prob.size: [N, 1]
    rotate_reg.size: [N, 1]
    bbox_reg.size: [N, 4]
    """
    height, width, _ = img.shape

    cls_prob = F.sigmoid(cls_prob).data
    rotate_reg = rotate_reg.data * 45
    bbox_reg = bbox_reg.data
    
    binary_tensor = (cls_prob>=threshold)
    indexes = binary_tensor.nonzero()[:, 0]
    if indexes.numel() < 0: return []
    rectangles = torch.index_select(rectangles, 0, indexes)
    rectangles[:, 4] = torch.index_select(cls_prob, 0, indexes) #--replace score
    rotate_reg = torch.index_select(rotate_reg, 0, indexes)

    index_180 = torch.eq(rectangles[:, 5], 180).nonzero()
    index_negative_90 = torch.eq(rectangles[:, 5], -90).nonzero()
    index_positive_90 = torch.eq(rectangles[:, 5], 90).nonzero()
    index_0 = torch.eq(rectangles[:, 5], 0).nonzero()

    if index_180.numel()>0: #---angle = 180的情况
        index_180 = index_180[:,0]
        rectangle_tmp = torch.index_select(rectangles, 0, index_180) 
        rectangle_tmp[:, 1], rectangle_tmp[:, 3] = height - 1 - rectangle_tmp[:, 3], height-1- rectangle_tmp[:, 1]         
        rectangles[index_180] = rectangle_tmp

    if index_positive_90.numel()>0:
        index_positive_90 = index_positive_90[:, 0] 
        rectangle_tmp = torch.index_select(rectangles, 0, index_positive_90)
        rectangle_tmp = rectangle_tmp[:,[1, 0, 3, 2, 4, 5]]
        rectangles[index_positive_90] = rectangle_tmp
    
    if index_negative_90.numel() > 0:
        index_negative_90 = index_negative_90[:, 0]
        rectangle_tmp = torch.index_select(rectangles, 0, index_negative_90) 
        rectangle_tmp = rectangle_tmp[:, [1, 2, 3, 0, 4, 5]]
        rectangle_tmp[:, 1] = width - 1 - rectangle_tmp[:, 1]
        rectangle_tmp[:, 3] = width - 1 - rectangle_tmp[:, 3] 
        rectangles[index_negative_90] = rectangle_tmp

    #if index_0.numel()>0:
    #    index_0 = index_0[:, 0]
    #    rectangle[index_0] = torch.index_select(rotate_reg, 0, index_0)
    rectangles[:, 5] = rotate_reg 

    #-----deal [x1, y1, x2, y2]------
    bbox_reg = torch.index_select(bbox_reg, 0, indexes)
    w = (rectangles[:, 2] - rectangles[:, 0]).view(-1, 1) #---[N]--->[N, 1]
    h = (rectangles[:, 3] - rectangles[:, 1]).view(-1, 1)
    bbox_reg[:, 0::2] = torch.mul(bbox_reg[:, 0::2], w)
    bbox_reg[:, 1::2] = torch.mul(bbox_reg[:, 1::2], h)
    rectangles[:, :4] = rectangles[:, :4] + bbox_reg
    rectangles = rect2square(rectangles)

    #---现在换算到原来的空间-----
    if index_180.numel()>0:
        rectangle_tmp = torch.index_select(rectangles, 0, index_180)
        rectangle_tmp[:, 1], rectangle_tmp[:, 3] = height - 1 - rectangle_tmp[:, 3], height-1- rectangle_tmp[:, 1]
        rectangle_tmp[:, 5] = 180 - rectangle_tmp[:, 5]
        rectangles[index_180] = rectangle_tmp

    if index_positive_90.numel()>0:
        rectangle_tmp = torch.index_select(rectangles, 0, index_positive_90)
        rectangle_tmp = rectangle_tmp[:,[1, 0, 3, 2, 4, 5]]
        rectangle_tmp[:, 5] = 90 - rectangle_tmp[:, 5]
        rectangles[index_positive_90] = rectangle_tmp
      
    if index_negative_90.numel()>0:
        rectangle_tmp = torch.index_select(rectangles, 0, index_negative_90)
        rectangle_tmp = rectangle_tmp[:, [3, 0, 1, 2, 4, 5]] 
        rectangle_tmp[:, 0] = width - 1 - rectangle_tmp[:, 0]
        rectangle_tmp[:, 2] = width - 1 - rectangle_tmp[:, 2]
        rectangle_tmp[:, 5] = 90 + rectangle_tmp[:, 5]
        rectangles[index_negative_90] = rectangle_tmp
 
    
    rectangles[:, :2] = torch.clamp(rectangles[:, :2], min=0)
    rectangles[:, 2] = torch.clamp(rectangles[:, 2], max=width)
    rectangles[:, 3] = torch.clamp(rectangles[:, 3], max=height)

    index1 = (rectangles[:, 2] > rectangles[:, 0]) #---x2 > x1
    index2 = (rectangles[:, 3] > rectangles[:, 1]) #---y2 > y1
    index = (index1 & index2).nonzero().squeeze()
    rectangles = rectangles.index_select(0, index)
    rectangles = NMS(rectangles, 0.5, 'iou')
    return rectangles

def PadImg(img):
    row, col, _ = img.shape
    row = min(int(row * 0.2), 100)
    col = min(int(col * 0.2), 100)
    img = cv2.copyMakeBorder(img, row, row, col, col, cv2.BORDER_CONSTANT)    
    return img


