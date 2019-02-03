import torch
from torchvision import transforms
import cv2
import numpy as np
import types
from numpy import random

def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]

def jaccard_numpy(box_a, box_b):
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1]))
    area_b = ((box_b[2]-box_b[0]) *
              (box_b[3]-box_b[1]))
    union = area_a + area_b - inter
    return inter / union

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, img, boxes=None, labels=None):
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)
        return img, boxes, labels

class ConvertFromInts(object):
    def __call__(self, image, boxes=None, labels=None):
        return image.astype(np.float32), boxes, labels

class ToAbsoluteCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height
        return image, boxes, labels

class ToPercentCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height
        return image, boxes, labels

class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."
    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)
        return image, boxes, labels

class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta
    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, boxes, labels

class SwapChannels(object):
    def __init__(self, swaps):
        self.swaps = swaps
    def __call__(self, image):
        image = image[:, :, self.swaps]
        return image

class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))
    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image, boxes, labels

class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current
    def __call__(self, image, boxes=None, labels=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image, boxes, labels

class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."
    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image, boxes, labels

class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta
    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image, boxes, labels

class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()
    def __call__(self, image, boxes, labels):
        im = image.copy()
        im, boxes, labels = self.rand_brightness(im, boxes, labels)
        if random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im, boxes, labels = distort(im, boxes, labels)
        return self.rand_light_noise(im, boxes, labels)

class ToTensor(object):
    def __call__(self, cvimage, boxes=None, labels=None):
        return torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1), boxes, labels

class Expand(object):
    def __init__(self):
        self.mean = (0,0,0)
    def __call__(self, image, boxes, labels):
        if random.randint(2):
            return image, boxes, labels
        height, width, depth = image.shape
        ratio = random.uniform(1, 3)
        left = random.uniform(0, width*ratio - width)
        top = random.uniform(0, height*ratio - height)
        expand_image = np.zeros((int(height*ratio), int(width*ratio), depth), dtype=image.dtype)
        expand_image[:, :, :] = self.mean
        expand_image[int(top):int(top + height), int(left):int(left + width)] = image
        image = expand_image
        boxes = boxes.copy()
        boxes[:, :2] += (int(left), int(top))
        boxes[:, 2:] += (int(left), int(top))
        return image, boxes, labels

class RandomMirror(object):
    def __call__(self, image, boxes, classes):
        _, width, _ = image.shape
        if random.randint(2):
            image = image[:, ::-1]
            boxes = boxes.copy()
            boxes[:, 0::2] = width - boxes[:, 2::-2]
        return image, boxes, classes

class Resize(object):
    def __init__(self, size=256):
        self.size = size
    def __call__(self, image, boxes=None, labels=None):
        image = cv2.resize(image, (self.size, self.size))
        return image, boxes, labels

class Bgr2Yuv(object):
    def __call__(self, image, boxes=None, labels=None):
        # cv2.imwrite('rgb.jpg', image)
        yuv = image.copy().astype(np.float32)
        yuv[:,:,0] = 0.299*image[:,:,2] + 0.587*image[:,:,1] + 0.114*image[:,:,0]
        yuv[:,:,1] = 0.492*(image[:,:,0] - yuv[:,:,0]) + 128
        yuv[:,:,2] = 0.877*(image[:,:,2] - yuv[:,:,0]) + 128
        image = yuv
        # cv2.imwrite('yuv.jpg', image[:,:,(2,1,0)])
        '''
        # color trans image.dtype must be uint8
        if image.dtype == 'uint8':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            cv2.imwrite('yuv.jpg', image[:,:,(2,1,0)])
        else:
            print('error data type in Bgr2Yuv')
        '''
        return image, boxes, labels

class ValueTrans(object):
    def __call__(self, image, boxes=None, labels=None):
        image = image.astype(np.float32)
        image /= 255.0
        return image, boxes, labels

class RandomSampleCrop(object):
    def __init__(self):
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )
    def __call__(self, image, boxes=None, labels=None):
        height, width, _ = image.shape
        while True:
            # randomly choose a mode
            mode = random.choice(self.sample_options)
            if mode is None:
                return image, boxes, labels
            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')
            # max trails (50)
            for _ in range(50):
                current_image = image
                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)
                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue
                left = random.uniform(width - w)
                top = random.uniform(height - h)
                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left+w), int(top+h)])
                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap = jaccard_numpy(boxes, rect)
                # is min and max overlap constraint satisfied? if not try again
                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue
                # cut the crop from the image
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2], :]
                # keep overlap with gt box IF center in sampled patch
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])
                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])
                # mask in that both m1 and m2 are true
                mask = m1 * m2
                # have any valid boxes? try again if not
                if not mask.any():
                    continue
                # take only matching gt boxes
                current_boxes = boxes[mask, :].copy()
                # take only matching gt labels
                current_labels = labels[mask]
                # should we use the box left and top corner or the crop's
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2], rect[:2])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, :2] -= rect[:2]
                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:], rect[2:])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, 2:] -= rect[:2]
                return current_image, current_boxes, current_labels

class PadResize(object):
    def __init__(self, size=256):
        self.size = size
    def __call__(self, image, boxes=None, labels=None):
        h, w, c = image.shape
        if h == w:
            return cv2.resize(image, (self.size, self.size)), boxes, labels
        if h > w:
            maxw = h
            t = 0
            l = int((h-w)/2)
        else:
            maxw = w
            t = int((w-h)/2)
            l = 0
        expand_image = np.zeros((maxw, maxw, c), dtype=image.dtype)
        expand_image[t:t+h, l:l+w] = image
        image = expand_image
        maxw = float(maxw)
        boxes[:, :2] *= (w/maxw, h/maxw)
        boxes[:, 2:] *= (w/maxw, h/maxw)
        boxes[:, :2] += (l/maxw, t/maxw)
        boxes[:, 2:] += (l/maxw, t/maxw)
        return cv2.resize(image, (self.size, self.size)), boxes, labels


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


class Rotate(object):
    def __init__(self):
        self.angles = [15,30, 45,60,75,90]
    def __call__(self, image, boxes=None, labels=None):
        angle = random.choice(self.angles)
        #image = rotate_bound(image, angle) 
        boxes = boxes.tolist()
        image, boxes = rot_clock(image, angle, boxes)
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
        return image, boxes, labels
    

def hidden_half_face(image, bbox, p=0.1):
    """
    if prob < p, then we take left size of the face or the right side of the face,
            and padding anther part by 0
    image: single image read by cv2
    bbox: gt_bbox, [[x1,y1,x2,y2]], here we just allow one bbox in gt_bbox
    """
    if len(bbox)>=2: return image, bbox
    heigth, width, _ = image.shape
    a_bbox = bbox[0]
    x1, y1, x2, y2 = a_bbox
    cx = (x1+x2)/2
    cx = int(cx) #---人脸的中线

    flag = 1 if random.random()>0.5 else 0 #-1, 左边涂黑; 0,右边涂黑
    if flag:
        image[:,:cx] = 0
        bbox[0][0] = cx
    else:
        image[:,cx:] = 0
        bbox[0][2] = cx

    return image, bbox

def hidden_half_face_version2(image, bbox):
    """
    image: single image read by cv2
    bbox: gt_bbox, [[x1,y1,x2,y2]], here we just allow one bbox in gt_bbox
    在这里，我的遮挡思路是这样：我用和人脸一样大小的区域遮挡人脸的[0.3, 0.5]的范围
    这个区域，我想从[0,0,0], [255,255,255],[128,128,128]，每个通道都随机取数
    hidden_flag: 0 表示没有做hidden处理, 1 表示做了hidden 处理。我想要把这个参数传出去，因为
        在后面的增强变换中会有一个random_crop的操作，我希望这个random_crop只对没有hidden过的
        图片做处理
    """
    hidden_flag = False
    if len(bbox)>=2: return image, bbox, hidden_flag
    height, width, channel = image.shape
    a_bbox = bbox[0]
    x1, y1, x2, y2 = a_bbox
    orig_box_width = x2 - x1 + 1
    orig_box_height = y2 - y1 + 1
    if orig_box_width * 1.0 / width <= 0.2:#---我们不对很小的人脸做遮挡
        return image, bbox, hidden_flag
    if orig_box_height * 1.0 / height <= 0.2:
        return image, bbox, hidden_flag
    cx = (x1+x2)/2
    cx = int(cx) #---人脸的中线

    cy = int((y1 + y2)/2) #---竖直方向上的中线

    #pad_images = np.zeros((box_height, box_width, channel), dtype=np.uint8)
    p = random.random()
    if p >= 0.6:
        flag = 1 #---涂黑左边
    elif p >= 0.3:
        flag = 2 #----涂黑右边
    else:
        flag = 3 #---涂黑下边

    scale_width = np.random.randint(0, 5) * 1.0 / 10 + 1.0
    scale_height = np.random.randint(0, 5) * 1.0 / 10 + 1.0
    #p2 = random.random()
    #if p2 >= 0.75:
    #    pad_image = pad_image + 255 #----255
    #elif p2 >= 0.5:
    #    pad_image = pad_image + 128 #----[128,128,128]
    #elif p2 >=0.25:
    #    pad_image[:,:,0] += np.random.randint(255)
    #    pad_image[:,:,1] += np.random.randint(255)
    #    pad_image[:,:,2] += np.random.randint(255)
    #else:
    #    pass
    box_width = orig_box_width * scale_width
    box_height = orig_box_height * scale_height
    if flag==1: #---遮挡左边
        left_x1 = int( cx - orig_box_width * 0.5 * 0.5) #---遮挡这边的时候, 水平方向上至少遮挡左半边脸的0.5
        pad_x2 = np.random.randint(left_x1, cx) #---确定遮挡区域的x2
        pad_x1 = max(pad_x2 - box_width + 1,0)  #---确定遮挡区域的x1

        pad_y1_min = max(cy - box_height , 0)
        pad_y1_max = cy
        pad_y1 = np.random.randint(pad_y1_min, pad_y1_max)
        pad_y2 = min(pad_y1 + box_height, height - 1)
    elif flag==2: #----遮挡右边
        pad_x1_min = cx
        pad_x1_max = int(cx + orig_box_width * 0.5 * 0.5)
        pad_x1 = np.random.randint(pad_x1_min, pad_x1_max)
        pad_x2 = min(pad_x1 + box_width, width-1)

        pad_y1_min = max(cy - box_height, 0)
        pad_y1_max = cy
        pad_y1 = np.random.randint(pad_y1_min, pad_y1_max)
        pad_y2 = min(pad_y1 + box_height, height - 1)
    elif flag==3: #----遮挡下边
        pad_y1_min = cy
        pad_y1_max = int(orig_box_height * 0.5 * 0.5 + cy)
        pad_y1 = np.random.randint(pad_y1_min, pad_y1_max)
        pad_y2 = min(pad_y1 + box_height, height- 1)


        pad_x1_min = max(cx - box_width, 0)
        pad_x1_max = cx
        if pad_x1_max <= pad_x1_min:
            return image, bbox, hidden_flag
        pad_x1 = np.random.randint(pad_x1_min, pad_x1_max)
        pad_x2 = min(pad_x1 + box_width, width - 1)

    pad_y1 = int(pad_y1)
    pad_y2 = int(pad_y2)
    pad_x1 = int(pad_x1)
    pad_x2 = int(pad_x2)
    pad_image = image[pad_y1:pad_y2, pad_x1:pad_x2]
    p2 = random.random()
    if p2 >= 0.75:
        pad_image = 255 #----255
    elif p2 >= 0.5:
        pad_image = 128 #----[128,128,128]
    elif p2 >=0.25:
        pad_image[:,:,0] = np.random.randint(0, 255)
        pad_image[:,:,1] = np.random.randint(0, 255)
        pad_image[:,:,2] = np.random.randint(0, 255)
    else:
        pad_image = 0

    image[pad_y1:pad_y2, pad_x1:pad_x2] = pad_image
    hidden_flag = True
    return image, bbox, hidden_flag

class HiddenFace(object):
    def __call__(self, image, bbox=None, labels=None):
        image, bbox = hidden_half_face(image, bbox)
        return image, bbox

