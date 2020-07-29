import os
from glob import glob

import cv2
import numpy as np
import skimage.io
from PIL import Image
from PIL import ImageFile


def get_points(filename, img_info):
    w, h = Image.open(filename).size
    values = img_info.split(',')[1:9]

    values = [float(i) for i in values]

    x1 = int(values[0] * w)
    x2 = int(values[1] * w)
    x3 = int(values[2] * w)
    x4 = int(values[3] * w)

    y1 = int(values[4] * h)
    y2 = int(values[5] * h)
    y3 = int(values[6] * h)
    y4 = int(values[7] * h)

    x = [x1, x2, x3, x4]
    y = [y1, y2, y3, y4]

    return x, y


def dpot(a, b):
    return (a - b) ** 2


def adist(a, b):
    return np.sqrt(dpot(a[0], b[0]) + dpot(a[1], b[1]))


def max_distance(a1, a2, b1, b2):
    dist1 = adist(a1, a2)
    dist2 = adist(b1, b2)
    if int(dist2) < int(dist1):
        return int(dist1)
    else:
        return int(dist2)


def sort_points(pts):
    ret = np.zeros((4, 2), dtype="float32")
    sumF = pts.sum(axis=1)
    diffF = np.diff(pts, axis=1)
    ret[0] = pts[np.argmin(sumF)]
    ret[1] = pts[np.argmin(diffF)]
    ret[2] = pts[np.argmax(sumF)]
    ret[3] = pts[np.argmax(diffF)]
    return ret


def fix_perspective(image, pts):
    (tl, tr, br, bl) = pts
    maxW = max_distance(br, bl, tr, tl)
    maxH = max_distance(tr, br, tl, bl)
    dst = np.array([[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]], dtype="float32")
    transform = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl], dtype="float32"), dst)
    fixed = cv2.warpPerspective(image, transform, (maxW, maxH))
    fixed = cv2.cvtColor(fixed, cv2.COLOR_RGB2BGR)
    return fixed


ROOT_DIR = os.path.abspath("../")
ImageFile.LOAD_TRUNCATED_IMAGES = True

folder = 'cropped'
if not os.path.exists(folder):
    os.makedirs(folder)

for filename in glob(os.path.join(ROOT_DIR, 'lpr_training', '*.txt')):
    filename = filename[:-4]
    # print(filename)

    with open(filename + '.txt') as f:
        img_info = f.readline()

    image = skimage.io.imread(filename + '.jpg')
    x, y = get_points(filename + '.jpg', img_info)
    points = []

    for i in range(len(x)):
        points.append((x[i], y[i]))

    points = sort_points(np.array(points))
    plate = fix_perspective(image, points)

    print(folder + '/' + os.path.basename(filename) + '.jpg')
    cv2.imwrite(folder + '/' + os.path.basename(filename) + '.jpg', plate)
