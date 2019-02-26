import sys
import numpy as np
import cv2
import os
import numpy.random as npr
from utils import IoU, rotate_images
from utils import ensure_directory_exists
sys.path.insert(0, "..")
import tools_matrix_torch as tools
import pcn
import torch
from torch.autograd import Variable

# image_name, cls_label, face_up_label, bbox,
# cls_label: [-1, 0, 1]  ----1 positive; 0 negative; -1 suspect, not contribute
# face_up_label: [-1, 0, 1] ----1 up; 0 down; -1 not contribute

device_id = 2
threshold = [0.2, 0.6, 0.6]
pnet = pcn.Pnet()
pnet.load_state_dict(torch.load("../pnet/pnet_190218_iter_1449000_.pth"))
pnet.eval()
pnet = pnet.cuda(device_id)


IMAGE_SIZE=24
DEBUG = True
if DEBUG:
    target_image_dir = "plot_images"
    ensure_directory_exists(target_image_dir)

anno_file = "wider_face_train.txt"
im_dir = "/media/disk1/mengfanli/new-caffe-workplace/MTCNN_workplace/mtcnn-caffe_without_landmarks/prepare_data/WIDER_train/images"
pos_save_dir = "../pnet/12_6/positive_hardmining"
suspect_save_dir = "../pnet/12_6/suspect_hardmining"
neg_save_dir = '../pnet/12_6/negative_hardmining'
save_dir = "../pnet/12_6"

ensure_directory_exists(save_dir)
ensure_directory_exists(pos_save_dir)
ensure_directory_exists(neg_save_dir)
ensure_directory_exists(suspect_save_dir)

f1 = open(os.path.join(save_dir, 'pos_hardmining_12.txt'), 'w')
f2 = open(os.path.join(save_dir, 'neg_hardmining_12.txt'), 'w')
f3 = open(os.path.join(save_dir, 'suspect_hardmining_12.txt'), 'w')

p_idx = 0 # positive
n_idx = 0 # negative
d_idx = 0 # dont care
idx = 0
box_idx = 0

angle_vecs = list(range(-180, 180))
face_up = list(range(-65, 66))
face_down = list(range(-180, -114)) + list(range(115, 181))

for a_line in open(anno_file):
    a_line = a_line.strip()
    array = a_line.split()
    if len(array) <= 2: continue
    a_image_name = array[0].split("/")[-1]
    a_subdir = array[0].split("/")[-2]
    bboxes_pos = array[1:]
    bboxes_pos = [float(x) for x in bboxes_pos]
    bboxes_pos = np.array(bboxes_pos, dtype=np.float32).reshape(-1, 5)
    bboxes = bboxes_pos[:, :-1]
    pos_vec = bboxes_pos[:, -1]
    
    a_image_path = os.path.join(im_dir, a_subdir, a_image_name)   
    print(a_image_path)
    a_image = cv2.imread(a_image_path)

    #-----count the number of images----
    idx += 1
    if idx % 100==0:
        print(idx, "images done")
    #---------------------------------- 
    
    select_angle = np.random.choice(angle_vecs)  
    a_image, bboxes = rotate_images(a_image, bboxes, select_angle)
    height, width, channel = a_image.shape 

    if select_angle in face_up: #----faceing up or down
        face_up_down_label = 1
    elif select_angle in face_down:
        face_up_down_label = 0
    else:
        face_up_down_label = -1

    image_pad = a_image.copy()
    original_h, original_w, ch = image_pad.shape
    scales = tools.calculateScales(image_pad)
    rectangles = []
    for scale in scales:
        hs = int(original_h * scale)
        ws = int(original_w * scale)
        scale_image = cv2.resize(image_pad,(ws,hs)) / 255.0 #---resize and rescale
        scale_image = scale_image.transpose((2, 0, 1))
        scale_image = torch.from_numpy(scale_image.copy())
        scale_image = torch.unsqueeze(scale_image, 0) #----[1, 3, H, W]
        scale_image = Variable(scale_image).float().cuda(device_id)
        conv4_1, _, conv4_2  = pnet(scale_image)
        cls_prob = conv4_1[0][0].cpu().data #----[1, 1, h, w] ----> [h, w]; varible to torch.tensor
        roi = conv4_2[0].cpu().data #---[1,4, h, w] -----> [4, h, w]; variable to torch.tensor
        out_w, out_h = cls_prob.size()
        out_side = max(out_w, out_h)
        rectangle = tools.detect_face_12net(cls_prob, roi, out_side, 1.0/scale, original_w, original_h, threshold[0])
        rectangles.extend(rectangle)

    rectangles = torch.stack(rectangles, 0)
    rectangles = tools.NMS_torch(rectangles, 0.5, 'iou')
    rectangles = rectangles.numpy()

    for a_rect in rectangles:
        a_rect_x1, a_rect_y1, a_rect_x2, a_rect_y2 , confidence =  a_rect
        if max((a_rect_x2 - a_rect_x1), (a_rect_y2 - a_rect_y1)) < 40:
            continue
        a_rect_width = a_rect_x2 - a_rect_x1 + 1
        a_rect_height = a_rect_y2 - a_rect_y1 + 1

        # ignore small faces
        # in case the ground truth boxes of small faces are not accurate
        if max(a_rect_width, a_rect_height) < 40 or a_rect_x1 < 0 or a_rect_y1 < 0:
            continue


        iou = IoU(a_rect, bboxes)
        max_id = np.argmax(iou)
        x1, y1, x2, y2 = bboxes[max_id]
        pos_flag = int(pos_vec[max_id])
        if pos_flag ==1: 
            face_up_down_label = -1 
        offset_x1 = (x1 - a_rect_x1) / float(a_rect_width)
        offset_y1 = (y1 - a_rect_y1) / float(a_rect_height)
        offset_x2 = (x2 - a_rect_x2) / float(a_rect_width)
        offset_y2 = (y2 - a_rect_y2) / float(a_rect_height)
        #print(a_rect_y1, a_rect_y2, a_rect_x1, a_rect_x2)
        cropped_im = a_image[int(a_rect_y1) : int(a_rect_y2), int(a_rect_x1) : int(a_rect_x2), :]
        resized_im = cv2.resize(cropped_im, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)

        if np.max(iou) >= 0.65:
            save_file = os.path.join(pos_save_dir, "%s.jpg"%p_idx)
            f1.write("12/positive_hardmining/%s.jpg"%p_idx + ' 1 %s %.4f %.4f %.4f %.4f\n'%(face_up_down_label, offset_x1, offset_y1, offset_x2, offset_y2))
            print("12/positive_hardmining/%s.jpg"%p_idx + ' 1 %s %.4f %.4f %.4f %.4f'%(face_up_down_label, offset_x1, offset_y1, offset_x2, offset_y2))
            print(" 12/positive_hardmining/%s.jpg"%p_idx + ' 1 %s %.4f %.4f %.4f %.4f'%(select_angle, offset_x1, offset_y1, offset_x2, offset_y2))
            cv2.imwrite(save_file, resized_im)
            p_idx += 1
        elif np.max(iou) >=0.4:
            if confidence >= 0.5:
                save_file = os.path.join(suspect_save_dir, "%s.jpg"%d_idx)
                f3.write("12/suspect_hardmining/%s.jpg"%d_idx + ' -1 %s %.4f %.4f %.4f %.4f\n'%(face_up_down_label, offset_x1, offset_y1, offset_x2, offset_y2))
                print("12/suspect_hardmining/%s.jpg"%d_idx + ' -1 %s %.4f %.4f %.4f %.4f'%(face_up_down_label, offset_x1, offset_y1, offset_x2, offset_y2))
                print(" 12/suspect_hardmining/%s.jpg"%d_idx + ' -1 %s %.4f %.4f %.4f %.4f'%(select_angle, offset_x1, offset_y1, offset_x2, offset_y2))
                cv2.imwrite(save_file, resized_im)
                d_idx += 1
f1.close()
f2.close()
f3.close()
