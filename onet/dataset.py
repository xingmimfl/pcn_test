# -*- coding: UTF-8 -*-
import cv2
import os
import shutil
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from augmentations import *
import glob
from config import *

#---a better augment is https://github.com/amdegroot/ssd.pytorch/blob/master/utils/augmentations.py

class ImageSets(Dataset):
    def __init__(self, isTrain=True, data_path=DATA_PATH, imageSize=48, 
            files_vec = None, images_vec = None):
        """
        files_vec = [pos_12.txt, neg_12.txt, part_12.txt]
        images_vec = [positive, negative, part] 
        """
        self.imageSize = imageSize
        self.isTrain = isTrain

        images_path_vec = []
        cls_labels_vec = []
        angle_labels_vec = []
        bbox_vec = []
        for a_file in files_vec:
            a_file_path = os.path.join(data_path, a_file)
            for a_line in open(a_file_path):
                a_line = a_line.strip()
                array = a_line.split()
                a_image_path = array[0]
                a_label = [int(float(array[1]))]
                a_angle_label = [float(array[2]) / 45.0]
                a_box = [float(x) for x in array[3:]]

                images_path_vec.append(a_image_path)
                cls_labels_vec.append(a_label)
                angle_labels_vec.append(a_angle_label)
                bbox_vec.append(a_box)

        self.indexes = [x for x in zip(images_path_vec, cls_labels_vec, angle_labels_vec, bbox_vec)] 
        random.shuffle(self.indexes)  
        #self.pts_indexes = pts_indexes #---vector of image name
        #self.images_indexes = images_indexes #---vector of image name

        self.augment = Compose([
                ConvertFromInts(),
                PhotometricDistort(),
                #Resize(imageSize),
                ValueTrans()
            ])

    def __len__(self):
        return len(self.indexes) 


    def __getitem__(self, index):
        a_image_path, a_label, a_angle_label, a_bbox = self.indexes[index]
        image = cv2.imread(a_image_path) #----be caution, we just consider jpg here
        a_bbox = np.asarray(a_bbox) #---turn bbox into numpy array
        a_label = np.asarray(a_label)  #----turn label into numpy array
        a_angle_label = np.asarray(a_angle_label) #---turn angle label into numpy array

        if self.isTrain:
            image, a_bbox, a_label = self.augment(image, a_bbox, a_label)   

        image = image.transpose((2, 0, 1)) #---[H, W, C] ----> [C, H, W]

        image = torch.from_numpy(image).float()
        a_bbox = torch.from_numpy(a_bbox).float() #---turn into tensor
        a_label = torch.from_numpy(a_label).int()
        a_angle_label = torch.from_numpy(a_angle_label).float()
        return image, a_bbox, a_label, a_angle_label, a_image_path

def detection_collate(batch):
    imgs = []
    targets = []
    labels = []
    angle_labels = []
    image_paths = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(sample[1])
        labels.append(sample[2]) 
        angle_labels.append(sample[3])
        image_paths.append(sample[4])
    return torch.stack(imgs, 0), torch.stack(targets, 0), torch.stack(labels, 0), \
        torch.stack(angle_labels, 0), image_paths


if __name__=="__main__":
    BATCH_SIZE = 2
    NUM_WORKERS = 1
    trainset = ImageSets(isTrain=True, data_path='.', imageSize=256, files_vec=['celeba_label.txt'], images_vec=None) 
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, collate_fn=detection_collate, \
                    num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)

    target_dir = "tmp_dataset_images"
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    for i_batch, sample_batched in enumerate(train_loader):
        images = sample_batched[0].numpy()
        bboxes = sample_batched[1].numpy()
        labels = sample_batched[2].numpy()
        angle_labels = sample_batched[3].numpy()
        image_paths = sample_batched[4]
        
        for i in range(BATCH_SIZE):
            a_image = images[i].transpose((1, 2, 0)) * 255
            a_image = a_image.astype(np.uint8).copy()
            height, width, channel = a_image.shape
            bbox = bboxes[i]
            bbox[0] = width * bbox[0];
            bbox[1] = height * bbox[1]
            bbox[2] = width * bbox[2]
            bbox[3] = height * bbox[3]             
            a_image_path = image_paths[i]
            a_image_name = os.path.basename(a_image_path)
            x1, y1, x2, y2 = bbox
            x1 = int(x1); y1 = int(y1)
            x2 = int(x2); y2 = int(y2)
            cv2.rectangle(a_image, (x1, y1), (x2, y2), (0,255,0),3)             

            a_image_target_path = os.path.join(target_dir, a_image_name)
            cv2.imwrite(a_image_target_path, a_image)
