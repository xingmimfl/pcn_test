import sys
import numpy as np
import cv2
import os
import numpy.random as npr
from utils import IoU, rotate_images
from utils import ensure_directory_exists

# image_name, cls_label, face_up_label, bbox,
# cls_label: [-1, 0, 1]  ----1 positive; 0 negative; -1 suspect, not contribute
# face_up_label: [-1, 0, 1] ----1 up; 0 down; -1 not contribute

IMAGE_SIZE=24
DEBUG = False
if DEBUG:
    target_image_dir = "plot_images"
    ensure_directory_exists(target_image_dir)
    
    negative_image_dir = "negative_plot_images"
    ensure_directory_exists(negative_image_dir)


anno_file = "wider_face_train.txt"
im_dir = "/media/disk1/mengfanli/new-caffe-workplace/MTCNN_workplace/mtcnn-caffe_without_landmarks/prepare_data/WIDER_train/images"
pos_save_dir = "../pnet/12/positive"
suspect_save_dir = "../pnet/12/suspect"
neg_save_dir = '../pnet/12/negative'
save_dir = "../pnet/12"

ensure_directory_exists(save_dir)
ensure_directory_exists(pos_save_dir)
ensure_directory_exists(neg_save_dir)
ensure_directory_exists(suspect_save_dir)

f1 = open(os.path.join(save_dir, 'pos_12.txt'), 'w')
f2 = open(os.path.join(save_dir, 'neg_12.txt'), 'w')
f3 = open(os.path.join(save_dir, 'suspect_12.txt'), 'w')

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
    
    neg_num = 0
    while neg_num < 100:
        a_image_copy = a_image.copy()
        bboxes_copy = bboxes.copy()
        select_angle = np.random.choice(angle_vecs)
        a_image_copy, bboxes_copy = rotate_images(a_image_copy, bboxes_copy, select_angle)
        height, width, channel = a_image_copy.shape

        size = npr.randint(40, min(width, height) / 2)
        nx = npr.randint(0, width - size)
        ny = npr.randint(0, height - size)
        crop_box = np.array([nx, ny, nx + size, ny + size])

        Iou = IoU(crop_box, bboxes_copy)

        cropped_im = a_image_copy[ny : ny + size, nx : nx + size, :].copy()
        resized_im = cv2.resize(cropped_im, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)

        if np.max(Iou) < 0.3:
            # Iou with all gts must below 0.3
            save_file = os.path.join(neg_save_dir, "%s.jpg"%n_idx)
            f2.write("12/negative/%s.jpg"%n_idx + ' 0 -1 -1 -1 -1 -1\n')
            cv2.imwrite(save_file, resized_im)
            n_idx += 1
            neg_num += 1

