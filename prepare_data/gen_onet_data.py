import sys
import numpy as np
import cv2
import os
import numpy.random as npr
from utils import IoU, rotate_images
from utils import ensure_directory_exists

# image_name, cls_label, face_up_label, bbox,
# cls_label: [-1, 0, 1]  ----1 positive; 0 negative; -1 suspect, not contribute
# face_label: [-45, 45]

IMAGE_SIZE=48
DEBUG = True
if DEBUG:
    target_image_dir = "plot_images"
    ensure_directory_exists(target_image_dir)

anno_file = "wider_face_train.txt"
im_dir = "/media/disk1/mengfanli/new-caffe-workplace/MTCNN_workplace/mtcnn-caffe_without_landmarks/prepare_data/WIDER_train/images"
pos_save_dir = "../onet/48/positive"
suspect_save_dir = "../onet/48/suspect"
neg_save_dir = '../onet/48/negative'
save_dir = "../onet/48"

ensure_directory_exists(save_dir)
ensure_directory_exists(pos_save_dir)
ensure_directory_exists(neg_save_dir)
ensure_directory_exists(suspect_save_dir)

f1 = open(os.path.join(save_dir, 'pos_48.txt'), 'w')
f2 = open(os.path.join(save_dir, 'neg_48.txt'), 'w')
f3 = open(os.path.join(save_dir, 'suspect_48.txt'), 'w')

p_idx = 0 # positive
n_idx = 0 # negative
d_idx = 0 # dont care
idx = 0
box_idx = 0

angle_vecs = list(range(-45, 46))

for a_line in open(anno_file):
    a_line = a_line.strip()
    array = a_line.split()
    if len(array) <= 2: continue
    a_image_name = array[0].split("/")[-1]
    a_subdir = array[0].split("/")[-2]
    bboxes = array[1:]
    bboxes = [float(x) for x in bboxes]
    bboxes = np.array(bboxes, dtype=np.float32).reshape(-1, 4)

    bboxes[:, 2] = (bboxes[:, 2] + bboxes[:, 3]) / 2.0 #---turn into square rectangles according to the paper
    bboxes[:, 3] = bboxes[:, 2]

    bboxes[:, 2] = bboxes[:, 2] + bboxes[:, 0] - 1 #---convert to x1, y1, x2, y2
    bboxes[:, 3] = bboxes[:, 3] + bboxes[:, 1] - 1
   
    a_image_path = os.path.join(im_dir, a_subdir, a_image_name)   
    print(a_image_path)
    a_image = cv2.imread(a_image_path)

    #-----count the number of images----
    idx += 1
    if idx % 100==0:
        print(idx, "images done")
    #---------------------------------- 
    
    height, width, channel = a_image.shape 
    #if DEBUG:
    #    for a_box in bboxes:
    #        x1, y1, w, h = a_box
    #        x2 = x1 + w - 1
    #        y2 = y1 + h - 1    
    #        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    #        cv2.rectangle(a_image, (x1, y1), (x2, y2), (0, 255, 0), 1)
    #        
    #    a_target_image_path = os.path.join(target_image_dir, a_image_name)
    #    cv2.imwrite(a_target_image_path, a_image)        

    select_angle = np.random.choice(angle_vecs)  
    a_image, bboxes = rotate_images(a_image, bboxes, select_angle)

    #if DEBUG:
    #    for a_box in bboxes:
    #        x1, y1, x2, y2 = a_box
    #        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    #        cv2.rectangle(a_image, (x1, y1), (x2, y2), (0, 255, 0), 1)
    #    a_target_image_path = os.path.join(target_image_dir, a_image_name)
    #    cv2.imwrite(a_target_image_path, a_image)    
    
    neg_num = 0
    while neg_num < 50:
        size = npr.randint(40, min(width, height) / 2)
        nx = npr.randint(0, width - size)
        ny = npr.randint(0, height - size)
        crop_box = np.array([nx, ny, nx + size, ny + size])

        Iou = IoU(crop_box, bboxes)

        cropped_im = a_image[ny : ny + size, nx : nx + size, :].copy()
        if cropped_im.sum() <=0: continue
        resized_im = cv2.resize(cropped_im, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)

        if np.max(Iou) < 0.3:
            # Iou with all gts must below 0.3
            save_file = os.path.join(neg_save_dir, "%s.jpg"%n_idx)
            f2.write("48/negative/%s.jpg"%n_idx + ' 0 -1 -1 -1 -1 -1\n')
            cv2.imwrite(save_file, resized_im)
            n_idx += 1
            neg_num += 1

    for box in bboxes:
        # box (x_left, y_top, x_right, y_bottom)
        x1, y1, x2, y2 = box
        w = x2 - x1 + 1
        h = y2 - y1 + 1

        # ignore small faces
        # in case the ground truth boxes of small faces are not accurate
        if max(w, h) < 48 or x1 < 0 or y1 < 0:
            continue

        # generate positive examples and suspect faces
        for i in range(20):
            size = npr.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))

            # delta here is the offset of box center
            delta_x = npr.randint(-w * 0.2, w * 0.2)
            delta_y = npr.randint(-h * 0.2, h * 0.2)

            nx1 = max(x1 + w / 2 + delta_x - size / 2, 0)
            ny1 = max(y1 + h / 2 + delta_y - size / 2, 0)
            nx2 = nx1 + size
            ny2 = ny1 + size

            if nx2 > width or ny2 > height:
                continue
            crop_box = np.array([nx1, ny1, nx2, ny2])

            offset_x1 = (x1 - nx1) / float(size)
            offset_y1 = (y1 - ny1) / float(size)
            offset_x2 = (x2 - nx2) / float(size)
            offset_y2 = (y2 - ny2) / float(size)

            cropped_im = a_image[int(ny1) : int(ny2), int(nx1) : int(nx2), :]
            resized_im = cv2.resize(cropped_im, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)

            box_ = box.reshape(1, -1)
            if IoU(crop_box, box_) >= 0.7:
                save_file = os.path.join(pos_save_dir, "%s.jpg"%p_idx)
                f1.write("48/positive/%s.jpg"%p_idx + ' 1 %.2f %.2f %.2f %.2f %.2f\n' % (select_angle, offset_x1, offset_y1, offset_x2, offset_y2))
                cv2.imwrite(save_file, resized_im)
                p_idx += 1
            elif IoU(crop_box, box_) >= 0.4:
                save_file = os.path.join(suspect_save_dir, "%s.jpg"%d_idx)
                f3.write("48/suspect/%s.jpg"%d_idx + ' -1 %.2f %.2f %.2f %.2f %.2f\n' % (select_angle, offset_x1, offset_y1, offset_x2, offset_y2))
                cv2.imwrite(save_file, resized_im)
                d_idx += 1
        box_idx += 1
        print("%s images done, pos: %s suspect: %s neg: %s"%(idx, p_idx, d_idx, n_idx))
    
