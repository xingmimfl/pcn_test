import os
import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class Pnet(nn.Module):
    def __init__(self):
        super(Pnet, self).__init__() 

        self.basenet = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=0, bias=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=0, bias=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=0, bias=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=1, padding=0,  bias=True),
            nn.ReLU(),            
        )
        self.fc5_1 = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc6_1 = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)
        self.bbox_reg_1 = nn.Conv2d(in_channels=128, out_channels=4, kernel_size=1, stride=1, padding=0, bias=True)
        
        self.cls_func = nn.BCELoss()
        self.angle_func = nn.BCELoss()
        self.bbox_func = nn.MSELoss()

    def forward(self, x):
        x = self.basenet(x)
        fc5 = self.fc5_1(x)
        fc5 = F.sigmoid(fc5)
        fc6 = self.fc6_1(x)
        fc6 = F.sigmoid(fc6)
        bbox_reg = self.bbox_reg_1(x) 
        return fc5, fc6, bbox_reg

    def get_loss(self, fc5, fc6, bbox_reg, cls_labels, angle_labels, bbox_labels):
        """
        fc5: for classification
        fc6: for angle classification
        """
        if len(fc5.size()) == 4:
            fc5 = fc5[:, :, 0, 0]
        if len(fc6.size()) == 4:
            fc6 = fc6[:, :, 0, 0]
        if len(bbox_reg.size()) == 4:
            bbox_reg = bbox_reg[:, :, 0, 0]

        loss_cls = None; loss_angle = None; loss_bbox = None
        #------cls loss-----
        positive_mask = torch.eq(cls_labels, 1)
        negative_mask = torch.eq(cls_labels, 0)
        cls_mask = (positive_mask | negative_mask) 
        cls_index = cls_mask.nonzero()
       
        if cls_index.numel() > 0:
            cls_index = cls_index[:, 0] 
            cls_labels_select = torch.index_select(cls_labels, 0, cls_index).float()
            fc5_select = torch.index_select(fc5, 0, cls_index)            
            loss_cls = self.cls_func(fc5_select, cls_labels_select)

        #------calibration loss---
        angle_pos_mask = torch.eq(angle_labels, 1)        
        angle_neg_mask = torch.eq(angle_labels, 0)
        angle_mask = (angle_pos_mask | angle_neg_mask)
        angle_index = angle_mask.nonzero()

        if angle_index.numel():
            angle_index = angle_index[:, 0]
            angle_labels_select = torch.index_select(angle_labels, 0, angle_index).float()
            #angle_labels_select = torch.squeeze(angle_labels_select)
            fc6_select = torch.index_select(fc6, 0, angle_index) 
            loss_angle = self.angle_func(fc6_select, angle_labels_select)

        #------bbox loss---
        suspect_mask = torch.eq(cls_labels, -1)
        bbox_mask = (positive_mask | suspect_mask) 
        bbox_index = bbox_mask.nonzero()
        
        if bbox_index.numel() > 0:
            bbox_index = bbox_index[:, 0]
            bbox_select = torch.index_select(bbox_labels, 0, bbox_index)
            bbox_reg_select = torch.index_select(bbox_reg, 0, bbox_index)
            loss_bbox = self.bbox_func(bbox_reg_select, bbox_select)
        
        return loss_cls, loss_angle, loss_bbox

class Rnet(nn.Module):
    def __init__(self):
        super(Rnet, self).__init__()
        self.basenet = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=20, kernel_size=3, stride=1, padding=0, bias=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=20, out_channels=40, kernel_size=3, stride=1, padding=0, bias=True), 
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            nn.ReLU(inplace=True), 
            
            nn.Conv2d(in_channels=40, out_channels=70, kernel_size=2, stride=1, padding=0, bias=True), 
            nn.ReLU(inplace=True),
        )
        
        self.fc4_2 = nn.Sequential(
            nn.Linear(in_features=70 * 3 * 3, out_features= 140),
            nn.ReLU(inplace=True),
        )
        self.fc5_2 = nn.Linear(in_features=140, out_features=1) #--class classification
        self.fc6_2 = nn.Linear(in_features=140, out_features=3) #---angle classification
        self.bbox_reg_2 = nn.Linear(in_features=140, out_features=4) #---bbox regression
         
        self.cls_func = nn.BCEWithLogitsLoss()
        self.angle_func = nn.CrossEntropyLoss()
        self.bbox_func = nn.MSELoss()

    def forward(self, x):
        x = self.basenet(x)
        x = x.view(-1, 70 * 3 * 3)
        x = self.fc4_2(x) 
        fc5_2 = self.fc5_2(x)
        fc6_2 = self.fc6_2(x)
        bbox_reg = self.bbox_reg_2(x)
        return fc5_2, fc6_2, bbox_reg

    def get_loss(self, fc5, fc6, bbox_reg, cls_labels, angle_labels, bbox_labels):
        #print(cls_labels)
        #print(angle_labels)
        if len(fc5.size()) == 4:
            fc5 = fc5[:, :, 0, 0]
        if len(fc6.size()) == 4:
            fc6 = fc6[:, :, 0, 0]
        if len(bbox_reg.size()) == 4:
            bbox_reg = bbox_reg[:, :, 0, 0]

        loss_cls = None; loss_angle = None; loss_bbox = None
        #------cls loss-----
        positive_mask = torch.eq(cls_labels, 1)
        negative_mask = torch.eq(cls_labels, 0)
        cls_mask = (positive_mask | negative_mask)
        cls_index = cls_mask.nonzero()

        if cls_index.numel() > 0:
            cls_index = cls_index[:, 0]
            cls_labels_select = torch.index_select(cls_labels, 0, cls_index).float()
            fc5_select = torch.index_select(fc5, 0, cls_index)
            loss_cls = self.cls_func(fc5_select, cls_labels_select)

        #------calibration loss---
        angle_0_mask = torch.eq(angle_labels, 0)
        angle_1_mask = torch.eq(angle_labels, 1)
        angle_2_mask = torch.eq(angle_labels, 2)
        angle_mask = (angle_0_mask | angle_1_mask | angle_2_mask)
        angle_index = angle_mask.nonzero()

        if angle_index.numel():
            angle_index = angle_index[:, 0]
            angle_labels_select = torch.index_select(angle_labels, 0, angle_index).squeeze()
            #print(angle_labels_select)
            fc6_select = torch.index_select(fc6, 0, angle_index)
            loss_angle = self.angle_func(fc6_select, angle_labels_select)

        #------bbox loss---
        suspect_mask = torch.eq(cls_labels, -1)
        bbox_mask = (positive_mask | suspect_mask)
        bbox_index = bbox_mask.nonzero()

        if bbox_index.numel() > 0:
            bbox_index = bbox_index[:, 0]
            bbox_labels_select = torch.index_select(bbox_labels, 0, bbox_index)
            #print(bbox_labels_select)
            bbox_reg_select = torch.index_select(bbox_reg, 0, bbox_index)
            loss_bbox = self.bbox_func(bbox_reg_select, bbox_labels_select)
        
        return loss_cls, loss_angle, loss_bbox

class Onet(nn.Module):
    def __init__(self):
        super(Onet, self).__init__()
        self.basenet = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=24, kernel_size=3, stride=1, padding=0, bias=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=24, out_channels=48, kernel_size=3, stride=1, padding=0, bias=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels=48, out_channels=96, kernel_size=3, stride=1, padding=0, bias=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=96, out_channels=144, kernel_size=2, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
        )
      
        self.fc5_3 = nn.Sequential(
            nn.Linear(in_features=144 * 3 * 3, out_features=192),
            nn.ReLU(inplace=True),
        ) 

        self.fc6_3 = nn.Linear(in_features=192, out_features=1) 
        self.bbox_reg_3 = nn.Linear(in_features=192, out_features=4)
        self.rotate_reg_3 = nn.Linear(in_features=192, out_features=1)
        
        self.cls_func = nn.BCEWithLogitsLoss()
        self.angle_func = nn.MSELoss()
        self.bbox_func = nn.MSELoss()

    def forward(self, x):
        x = self.basenet(x)
        x = x.view(-1, 144 * 3 * 3)
        x = self.fc5_3(x)
        fc6 = self.fc6_3(x)
        bbox_reg = self.bbox_reg_3(x)
        rotate_reg = self.rotate_reg_3(x)
        return fc6, rotate_reg, bbox_reg

    #def get_loss(self, fc6_cls, rotate_reg, bbox_reg, cls_labels, angle_labels, bbox_labels):
    def get_loss(self, fc5, fc6, bbox_reg, cls_labels, angle_labels, bbox_labels):
        #print(angle_labels)
        #print(cls_labels)
        #print(bbox_labels)
        #fc5: classifcation labels
        #fc6: angle regression
        if len(fc5.size()) == 4:
            fc5 = fc5[:, :, 0, 0]
        if len(fc6.size()) == 4:
            fc6 = fc6[:, :, 0, 0]
        if len(bbox_reg.size()) == 4:
            bbox_reg = bbox_reg[:, :, 0, 0]

        loss_cls = None; loss_angle = None; loss_bbox = None
        #------cls loss-----
        positive_mask = torch.eq(cls_labels, 1)
        negative_mask = torch.eq(cls_labels, 0)
        cls_mask = (positive_mask | negative_mask)
        cls_index = cls_mask.nonzero()

        if cls_index.numel() > 0:
            cls_index = cls_index[:, 0]
            cls_labels_select = torch.index_select(cls_labels, 0, cls_index).float()
            fc5_select = torch.index_select(fc5, 0, cls_index)
            loss_cls = self.cls_func(fc5_select, cls_labels_select)

        #------bbox loss---
        suspect_mask = torch.eq(cls_labels, -1)
        bbox_mask = (positive_mask | suspect_mask)
        bbox_index = bbox_mask.nonzero()        

        if bbox_index.numel() > 0:
            bbox_index = bbox_index[:, 0]
            bbox_labels_select = torch.index_select(bbox_labels, 0, bbox_index)
            bbox_reg_select = torch.index_select(bbox_reg, 0, bbox_index)
            loss_bbox = self.bbox_func(bbox_reg_select, bbox_labels_select)

        #------calibration loss---
        angle_mask = (positive_mask | suspect_mask)
        angle_index = angle_mask.nonzero()

        if angle_index.numel():
            angle_index = angle_index[:, 0]
            angle_labels_select = torch.index_select(angle_labels, 0, angle_index).float()
            fc6_select = torch.index_select(fc6, 0, angle_index)
            loss_angle = self.angle_func(fc6_select, angle_labels_select)

        return loss_cls, loss_angle, loss_bbox

if __name__=="__main__":
    #x = torch.rand((64, 3, 24, 24))
    x = torch.rand((64, 3, 100, 100))
    x = Variable(x)
    net = Pnet()
    fc5, fc6, bbox_reg = net(x)
    print("fc5.size:\t", fc5.size())
    print("fc6.size:\t", fc6.size())
    print("bbox_reg.size:\t", bbox_reg.size())
    
