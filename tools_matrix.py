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
        imgResized_input = Variable(imgResized_input).float().cuda(device_id)
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
            print(rotate_score)
            rotate_positive_mask = (rotate_score >= 0.5)
            print(rotate_positive_mask)
            rotate_negative_mask = ~rotate_positive_mask
            rotate_score[rotate_positive_mask] = 180.0
            rotate_score[rotate_negative_mask]  = 0.
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
                rectangles = rectangles.data.cpu()
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
    prob = prob.data.cpu(); #---variable to tensor
    angle_prob = angle_prob.data.cpu(); #---variable to tensor
    bbox_reg = bbox_reg.data.cpu() #---variable to tensor

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
    print(angle_max_index)
    angle_max_index_1 = torch.eq(angle_max_index, 1)    #---max_index is 1
    rectangles = torch.index_select(rectangles, 0, indexes) #----select rectangles of positive samples
    rectangles[:, 4] = torch.index_select(prob, 0, indexes) #----replace classification score

    
    face_down_mask = torch.eq(rectangles[:, 5], 0) #----mask for face down
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
        rectangles_up[:, 5] = (angle_max_index_up - 1) * -90
        angle_max_index_down_1 = (angle_max_index_1 & face_down_index)
        rectangles_up[angle_max_index_down_1, 5] = 180
        #rectangles_down[:, 1], rectangles_down[:, 3] = torch.add(height-1, rectangles_down[:, 3] * -1), torch.add(height-1, rectangles_down[:, 1] * -1) #---img180中的地址
        rectangles_down[:, 1], rectangles_down[:, 3] = height-1 - rectangles_down[:, 3], height-1 - rectangles_down[:, 1] #---img180中的坐标
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
        rectangles_down[face_down_index, 1], rectangles_down[face_down_index, 3] = height-1 - rectangles_down[face_down_index, 3],\
                 height-1 - rectangles_down[face_down_index, 1] 
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

def filter_face_onet(cls_prob, rotate_reg, bbox_reg, rectangles, imgPad, threshold):
    """
    cls_prob.size: [N, 1]
    rotate_reg.size: [N, 1]
    bbox_reg.size: [N, 4]
    """
    binary_tensor = (cls_prob>=threshold)
    indexes = binary_tensor.nonzero()[:, 0]

    rectangles = torch.index_select(rectangles, 0, indexes)
    rectangles[:, 4] = torch.index_select(cls_prob, 0, indexes) #--replace score

    bbox_reg = torch.index_select(bbox_reg, 0, indexes)
    w = (rectangles[:, 2] - rectangles[:, 0]).view(-1, 1)
    h = (rectangles[:, 3] - rectangles[:, 1]).view(-1, 1)

    bbox_reg[:, 0::2] = torch.mul(bbox_reg[:, 0::2], w)
    bbox_reg[:, 1::2] = torch.mul(bbox_reg[:, 1::2], h)
    rectangles[:, :4] = rectangles[:, :4] + bbox_reg

          
    """
    pts = pts.index_select(0, indexes)
    pts[:, 0::2] = torch.mul(pts[:, 0::2], w)
    pts[:, 0::2] = torch.add(pts[:, 0::2], rectangles[:, 0].view(-1,1)) 
    pts[:, 1::2] = torch.mul(pts[:, 1::2], h)
    pts[:, 1::2] = torch.add(pts[:, 1::2], rectangles[:, 1].view(-1,1))

    #-----
    rectangles[:, :2] = torch.clamp(rectangles[:, :2], min=0)
    rectangles[:, 2] = torch.clamp(rectangles[:, 2], max=width)
    rectangles[:, 3] = torch.clamp(rectangles[:, 3], max=height)

    index1 = (rectangles[:, 2] >= rectangles[:, 0]) #---x2 > x1
    index2 = (rectangles[:, 3] >= rectangles[:, 1]) #---y2 > y1
    
    #---concat rectangles and pts
    rectangles = torch.cat([rectangles, pts], dim=1) 
    #rectangles = rectangles.numpy()
    rectangles = NMS_torch(rectangles, 0.7,'iom')
    """
    return rectangles

def PadImg(img):
    row, col, _ = img.shape
    row = min(int(row * 0.2), 100)
    col = min(int(col * 0.2), 100)
    img = cv2.copyMakeBorder(img, row, row, col, col, cv2.BORDER_CONSTANT)    
    return img


