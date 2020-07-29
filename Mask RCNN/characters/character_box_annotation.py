import json
import os
from glob import glob

import character_segmentation
import cv2
import numpy as np

line = ''
license_plate = {}
data_train = {}
data_val = {}

with open('model_data/character_classes.txt', 'r') as f:
    characters = f.read().splitlines()

for tmp_line in open('../../Datasets/training_LP.txt', 'r'):
    tmp_line = tmp_line.strip().split('/')
    license_plate[tmp_line[0]] = tmp_line[1]

for filename in glob(os.path.join('../../Datasets/cropped', '*.jpg')):
    img = cv2.imread(filename)
    h, w = img.shape[:2]

    if w > 2 * h:
        img = cv2.resize(img, (600, 150))
    else:
        img = cv2.resize(img, (300, 300))

    origin = img

    img = character_segmentation.clean_image(img)
    clean_img, chars = character_segmentation.extract_characters(img)

    image_name = os.path.basename(filename)
    print(image_name)
    save_image_path = '../../Datasets/train_cropped/' + image_name

    if len(chars) is 7:
        cv2.imwrite(save_image_path, origin)
        curr_plate = license_plate[image_name]

        img_data = {}
        img_data['filename'] = save_image_path
        regions = {}

        for i in range(len(chars)):
            x, y, w, h = chars[i][0]
            all_x = [x, x + w, x + w, x]
            all_y = [y, y, y+h, y+h]
            class_id = 1 + characters.index(curr_plate[i])

            regions[str(i)] = {'shape_attributes': {'name': 'polygon', 'all_points_x': all_x, 'all_points_y': all_y},
                               'region_attributes': {'characters': str(class_id)}}

        img_data['regions'] = regions

        ch = np.random.choice([0, 1], p=[0.85, 0.15])
        if ch:
            data_val[save_image_path] = img_data
        else:
            data_train[save_image_path] = img_data

with open('model_data/data_val_box.json', 'w') as outfile:
    json.dump(data_val, outfile)

with open('model_data/data_train_box.json', 'w') as outfile:
    json.dump(data_train, outfile)
