import os
import sys
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
sys.path.append("..")
from config import *
import dataset
import pcn

def main():
    with torch.cuda.device(DEVICE_IDS[0]):
        r_model = _model_init()
        #----data loader----
        #train_loader = get_dataset(files_vec=['pos_24.txt', 'neg_24.txt', 'suspect_24.txt'])
        train_loader = get_dataset(files_vec=['pos_24.txt', 'neg_24.txt', 'suspect_24.txt', 'pos_pnet_24.txt', 'suspect_pnet_24.txt','neg_pnet_24.txt'])
        train_iter = iter(train_loader)
        #----data iter-----
        check_dir(SNAPSHOT_PATH)

        #----get parameters that need back-propagation
        params = []
        for p in list(r_model.parameters()):
            if p.requires_grad == False: continue
            params.append(p)

        #----optimizer
        optimizer = optim.SGD(params, lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
        scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
        #----training
        loss_avg = AverageMeter()
        loss_cls_avg = AverageMeter()
        loss_bbox_avg = AverageMeter()
        loss_angle_avg = AverageMeter()
        acc1 = AverageMeter()

        for i in range(MAX_ITERS):
            scheduler.step() 
            #---load data----
            try:
                _images, _bbox, _labels, _angle_labels, _image_paths = train_iter.next()
            except StopIteration:
                train_iter = iter(train_loader)
                _images, _bbox, _labels, _angle_labels, _image_paths = train_iter.next() 
 
            #-----training, get loss, and back-propagation
            _images = Variable(_images.cuda(DEVICE_IDS[0]))
            _bbox = Variable(_bbox.cuda(DEVICE_IDS[0]))

            _labels = _labels.cuda(DEVICE_IDS[0])
            _labels_var = Variable(_labels)

            _angle_labels = _angle_labels.cuda(DEVICE_IDS[0])
            _angle_labels_var = Variable(_angle_labels)

            fc5, fc6, bbox_reg = r_model(_images) #----model forward 
            loss_cls, loss_angle, loss_bbox = r_model.get_loss(fc5, fc6, bbox_reg,\
                                                 _labels_var, _angle_labels_var, _bbox)

            loss = 0
            if loss_cls is not None:
                loss += loss_cls
                loss_cls_avg.update(loss_cls.data[0], BATCH_SIZE)
                
            if loss_bbox is not None:
                loss += loss_bbox
                loss_bbox_avg.update(loss_bbox.data[0], BATCH_SIZE)

            if loss_angle is not None:
                loss += loss_angle
                loss_angle_avg.update(loss_angle.data[0], BATCH_SIZE)

            prec1 =  accuracy(fc5.data, _labels) 
            acc1.update(prec1)

            loss_avg.update(loss.data[0], BATCH_SIZE)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i >=TRAIN_OUT_ITER)  and (i % TRAIN_OUT_ITER == 0):  # print every 2000 mini-batches
                print("iter:%5d " % i, " loss:%.4e" % loss_avg.avg, " loss_cls:%.4e" % loss_cls_avg.avg, 
                        " loss_bbox:%.4e" % loss_bbox_avg.avg, " loss_angle:%.4e" % loss_angle_avg.avg, " accuracy:%.4e" % acc1.avg)

                save_name = '_'.join([SUFFIX, "iter", str(i), '.model'])                
                torch.save(r_model, os.path.join(SNAPSHOT_PATH, save_name))

                loss_avg = AverageMeter()
                loss_cls_avg = AverageMeter()
                loss_bbox_avg = AverageMeter()
                loss_angle_avg = AverageMeter()
                acc1 = AverageMeter()
                 
def get_dataset(files_vec=None, images_vec=None):
    trainset = dataset.ImageSets(isTrain=True, imageSize=12, files_vec=files_vec, images_vec=images_vec)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, collate_fn=dataset.detection_collate, 
                                                num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    print("length of trainset:\t", len(trainset))
    return train_loader

     
def _model_init():
    r_model = pcn.Rnet()
    #r_model.apply(weight_init)
    r_model.cuda(DEVICE_IDS[0])
    r_model.train()
    return r_model


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target):
    #output = output[:,:,0,0] #---[batch_size, 1, 1, 1] ---> [batch_size, 1] 
    batch_size = output.size(0)
    target = target.long() 
    output = (output >= 0.5).type_as(target)
    correct = output.eq(target)
    res = correct.sum() * 100.0 / batch_size 
    return res

if __name__=="__main__":
    main()
