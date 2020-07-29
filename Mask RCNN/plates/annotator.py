import json
import os
from glob import glob

import numpy as np
from PIL import Image


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


data_train = {}
data_val = {}

for filename in glob(os.path.join('lpr_training', '*.txt')):
    print(filename)
    filename = filename[:-4]
    print(filename)
    with open(filename + '.txt') as f:
        img_info = f.readline()
        img_data = {}
        img_data['filename'] = filename + '.jpg'
        x, y = get_points(filename + '.jpg', img_info)
        regions = {'0': {'shape_attributes': {'name': 'polygon', 'all_points_x': x, 'all_points_y': y}}}
        img_data['regions'] = regions

        ch = np.random.choice([0, 1], p=[0.85, 0.15])
        if ch:
            data_val[filename] = img_data
        else:
            data_train[filename] = img_data

with open('Mask RCNN/plates/model_data/data_val.json', 'w') as outfile:
    json.dump(data_val, outfile)

with open('Mask RCNN/plates/model_data/data_train.json', 'w') as outfile:
    json.dump(data_train, outfile)
