import sys
import numpy as np
import cv2
import os
import numpy.random as npr
from utils import IoU, rotate_images
from utils import ensure_directory_exists
sys.path.insert(0, "..")
import tools_matrix_cpu as tools
import pcn
import torch
from torch.autograd import Variable

# image_name, cls_label, face_up_label, bbox,
# cls_label: [-1, 0, 1]  ----1 positive; 0 negative; -1 suspect, not contribute
# face_up_label: [-1, 0, 1] ----1 up; 0 down; -1 not contribute

device_id = 1
threshold = [0.2, 0.6, 0.6]
pnet = pcn.Pnet()
pnet.load_state_dict(torch.load("../pnet/pnet_190302_version2_iter_1239000_.pth"))
pnet.eval()


IMAGE_SIZE=24
DEBUG = True
if DEBUG:
    target_image_dir = "plot_images"
    ensure_directory_exists(target_image_dir)

anno_file = "wider_face_train.txt"
im_dir = "/media/disk1/mengfanli/new-caffe-workplace/MTCNN_workplace/mtcnn-caffe_without_landmarks/prepare_data/WIDER_train/images"
pos_save_dir = "../pnet/12/positive_hardmining"
suspect_save_dir = "../pnet/12/suspect_hardmining"
neg_save_dir = '../pnet/12/negative_hardmining'
save_dir = "../pnet/12"

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

count = 0
while count < 10:
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
        rectangles = tools.detect_face_pnet(a_image, image_pad, pnet, 0.5, device_id)
        rectangles = tools.NMS(rectangles, 0.5, 'iou')    

        for a_rect in rectangles:
            a_rect_x1, a_rect_y1, a_rect_x2, a_rect_y2 , confidence, _ =  a_rect
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

            if np.max(iou) >= 0.7:
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
            elif np.max(iou) < 0.3: #--negative samples
                if confidence >= 0.8:
                    save_file = os.path.join(neg_save_dir, "%s.jpg"%n_idx)
                    f2.write("12/negative_hardmining/%s.jpg"%n_idx + ' 0 -1 -1 -1 -1 -1\n')
                    cv2.imwrite(save_file, resized_im)
                    n_idx += 1
    count = count + 1


f1.close()
f2.close()
f3.close()
