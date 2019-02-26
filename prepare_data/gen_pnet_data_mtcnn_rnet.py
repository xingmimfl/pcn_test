import sys
import numpy as np
import cv2
import os
import numpy.random as npr
from utils import IoU, rotate_images
from utils import ensure_directory_exists

# image_name, cls_label, face_up_label, bbox,
# cls_label: [-1, 0, 1]  ----1 positive; 0 negative; -1 part, not contribute
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
pos_save_dir = "../mtcnn_rnet/24/positive"
part_save_dir = "../mtcnn_rnet/24/part"
neg_save_dir = '../mtcnn_rnet/24/negative'
save_dir = "../mtcnn_rnet/24"

ensure_directory_exists(save_dir)
ensure_directory_exists(pos_save_dir)
ensure_directory_exists(neg_save_dir)
ensure_directory_exists(part_save_dir)

f1 = open(os.path.join(save_dir, 'pos_24.txt'), 'w')
f2 = open(os.path.join(save_dir, 'neg_24.txt'), 'w')
f3 = open(os.path.join(save_dir, 'part_24.txt'), 'w')

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
    select_angle = 0
    height, width, channel = a_image.shape 

    if DEBUG:
        a_image_copy = a_image.copy()
        for a_box in bboxes:
            x1, y1, x2, y2 = a_box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(a_image_copy, (x1, y1), (x2, y2), (0, 255, 0), 1)
        a_target_image_path = os.path.join(target_image_dir, a_image_name)
        cv2.imwrite(a_target_image_path, a_image_copy)    
    
    if select_angle in face_up: #----faceing up or down
        face_up_down_label = 1
    elif select_angle in face_down:
        face_up_down_label = 0
    else:
        face_up_down_label = -1


    neg_num = 0
    if DEBUG:
        negative_image_path = os.path.join(negative_image_dir, a_image_name)
        negative_image = a_image.copy()

    while neg_num < 60:
        size = npr.randint(40, min(width, height) / 2)
        nx = npr.randint(0, width - size)
        ny = npr.randint(0, height - size)
        crop_box = np.array([nx, ny, nx + size, ny + size])

        Iou = IoU(crop_box, bboxes)

        cropped_im = a_image[ny : ny + size, nx : nx + size, :].copy()
        resized_im = cv2.resize(cropped_im, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)

        if DEBUG:
            cv2.rectangle(negative_image, (nx, ny), (nx + size, ny + size), (0, 255, 0), 2)

        if np.max(Iou) < 0.3:
            # Iou with all gts must below 0.3
            save_file = os.path.join(neg_save_dir, "%s.jpg"%n_idx)
            f2.write("24/negative/%s.jpg"%n_idx + ' 0 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1\n')
            cv2.imwrite(save_file, resized_im)
            n_idx += 1
            neg_num += 1

    if DEBUG:
        cv2.imwrite(negative_image_path, negative_image)
    
    num_of_bboxes = bboxes.shape[0]
    #for box in bboxes:
    for i in range(num_of_bboxes):
        box = bboxes[i]
        pos_value = int(pos_vec[i])
        if pos_value == 1:
            face_up_down_label = -1
        x1, y1, x2, y2 = box
        w = x2 - x1 + 1
        h = y2 - y1 + 1

        # ignore small faces
        # in case the ground truth boxes of small faces are not accurate
        if max(w, h) < 40 or x1 < 0 or y1 < 0:
            continue

        # generate positive examples and part faces
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
                f1.write("24/positive/%s.jpg"%p_idx + ' 1 %.4f %.4f %.4f %.4f' % (offset_x1, offset_y1, offset_x2, offset_y2) + " -1 -1 -1 -1 -1 -1\n")
                cv2.imwrite(save_file, resized_im)
                p_idx += 1
            elif IoU(crop_box, box_) >= 0.4:
                save_file = os.path.join(part_save_dir, "%s.jpg"%d_idx)
                f3.write("24/part/%s.jpg"%d_idx + ' -1 %.4f %.4f %.4f %.4f' % (offset_x1, offset_y1, offset_x2, offset_y2) + " -1 -1 -1 -1 -1 -1\n")
                cv2.imwrite(save_file, resized_im)
                d_idx += 1
            #elif IoU(crop_box, box_) < 0.3:
            #    save_file = os.path.join(neg_save_dir, "%s.jpg"%n_idx)
            #    f2.write("24/negative/%s.jpg"%n_idx + ' 0 -1 -1 -1 -1 -1\n$')
            #   cv2.imwrite(save_file, resized_im)
            #    n_idx += 1
            #    neg_num += 1

        box_idx += 1
        print("%s images done, pos: %s part: %s neg: %s"%(idx, p_idx, d_idx, n_idx))
    
