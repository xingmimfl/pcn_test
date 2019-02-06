import os
import cv2
import numpy as np

def IoU(box, boxes):
    """Compute IoU between detect box and gt boxes

    Parameters:
    ----------
    box: numpy array , shape (5, ): x1, y1, x2, y2, score
        input box
    boxes: numpy array, shape (n, 4): x1, y1, x2, y2
        input ground truth boxes

    Returns:
    -------
    ovr: numpy.array, shape (n, )
        IoU
    """
    box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])

    # compute the width and height of the bounding box
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)

    inter = w * h
    ovr = inter / (box_area + area - inter)
    return ovr


def convert_to_square(bbox):
    """Convert bbox to square

    Parameters:
    ----------
    bbox: numpy array , shape n x 5
        input bbox

    Returns:
    -------
    square bbox
    """
    square_bbox = bbox.copy()

    h = bbox[:, 3] - bbox[:, 1] + 1
    w = bbox[:, 2] - bbox[:, 0] + 1
    max_side = np.maximum(h,w)
    square_bbox[:, 0] = bbox[:, 0] + w*0.5 - max_side*0.5
    square_bbox[:, 1] = bbox[:, 1] + h*0.5 - max_side*0.5
    square_bbox[:, 2] = square_bbox[:, 0] + max_side - 1
    square_bbox[:, 3] = square_bbox[:, 1] + max_side - 1
    return square_bbox

def ensure_directory_exists(path):
    if not os.path.exists(path):
        os.mkdir(path)


def rotate_box(bb, cx, cy, h, w, angle):
    new_bb = []
    for i, coord in enumerate(bb):
        m = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)
        cos = np.abs(m[0, 0])
        sin = np.abs(m[0, 1])
        n_w = int((h * sin) + (w * cos))
        n_h = int((h * cos) + (w * sin))
        m[0, 2] += (n_w / 2) - cx
        m[1, 2] += (n_h / 2) - cy
        v = [coord[0], coord[1], 1]
        calculated = np.dot(m, v)
        new_bb.append(int(round(calculated[0], 0)))
        new_bb.append(int(round(calculated[1], 0)))
    return new_bb


def rotate_bound(img, angle):
    (h, w) = img.shape[:2]
    (c_x, c_y) = (w // 2, h // 2)

    m = cv2.getRotationMatrix2D((c_x, c_y), -angle, 1.0)
    cos = np.abs(m[0, 0])
    sin = np.abs(m[0, 1])

    n_w = int((h * sin) + (w * cos))
    n_h = int((h * cos) + (w * sin))

    m[0, 2] += (n_w / 2) - c_x
    m[1, 2] += (n_h / 2) - c_y

    return cv2.warpAffine(img, m, (n_w, n_h))

def rot_clock(img, angle, coords):
    bb = coords_to_bb(coords)
    rotated_img = rotate_bound(img, angle)

    (h, w) = img.shape[:2]
    (cx, cy) = (w // 2, h // 2)

    new_bb = []
    for i in bb:
        new_bb.append(rotate_box(i, cx, cy, h, w, angle))

    return rotated_img, new_bb


def coords_to_bb(coords):
    bb = []
    for c in coords:
        c[0] = float(c[0])
        c[1] = float(c[1])
        c[2] = float(c[2])
        c[3] = float(c[3])
        bb.append([(c[0], c[1]), (c[2], c[1]), (c[2], c[3]), (c[0], c[3])])

    return bb

def rotate_images(a_image, boxes, angle):
    """
    a_image: bgr image read by cv2
    bboxes: np.array [[x1, y1, x2, y2]]
    """
    boxes = boxes.tolist()
    image, boxes = rot_clock(a_image, angle, boxes)
    new_boxes = []
    x1=99999; y1=99999
    x2=0; y2=0
    for a_box in boxes:
        x1 = min([a_box[0], a_box[2], a_box[4], a_box[6]])
        x2 = max([a_box[0], a_box[2], a_box[4], a_box[6]])
        y1 = min([a_box[1], a_box[3], a_box[5], a_box[7]])
        y2 = max([a_box[1], a_box[3], a_box[5], a_box[7]])
        new_boxes.append([x1, y1, x2, y2])

    boxes = np.asarray(new_boxes).astype(np.float32)     
    return image, boxes 
