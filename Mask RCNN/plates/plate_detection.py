import argparse
import os
import sys

import cv2
import numpy as np
import tensorflow as tf

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils

import mrcnn.model as modellib

import plates


def get_corner_points(r):
    masks = r['masks']

    if masks.shape[2] == 0:
        return None
    # Si considera la maschera più grande
    index = np.argmax(np.sum(np.sum(masks, axis=1), axis=0))
    masks = np.moveaxis(masks, -1, 0)
    bigger_mask = masks[index]

    bigger_mask = np.array(bigger_mask * 255).astype('uint8')

    #Si effettua il threshold sulla maschera
    ret, thresh = cv2.threshold(bigger_mask, 127, 255, 0)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=3)

    # Si definiscono i contorni della targa
    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #Si estraggono i quattro vertici del poligono e si ordinano in senso orario
    epsilon = 0.03 * cv2.arcLength(contours[0], True)
    approx = cv2.approxPolyDP(contours[0], epsilon, True)

    p = [x[0] for x in approx]

    p = sort_points(np.array(p))

    return p


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


if __name__ == '__main__':

    # Directory to save logs and trained model
    MODEL_DIR = "model_data/weights/"

    # Path to Plates trained weights
    PLATES_WEIGHTS_PATH = MODEL_DIR + "mask_rcnn_plate_0042.h5"  # TODO: update this path

    parser = argparse.ArgumentParser(description='MASK R-CNN plate detector')
    parser.add_argument('--test_image_path', required=False,
                        metavar="/path/to/test_images",
                        help="Path to test images")
    parser.add_argument("--plates_path", required=True, type=str, nargs='?',
                        help="Path della cartella in cui salvare le targhe valutate.")
    parser.add_argument("--results_path", required=True, type=str, nargs='?',
                        help="Path della cartella in cui salvare i risultati.")
    parser.add_argument("--dataset", required=True, type=str, nargs='?',
                        help="train - test - val")
    args = parser.parse_args()

    plates_path = args.plates_path
    results_path = args.results_path

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    if not os.path.exists(plates_path):
        os.makedirs(plates_path)

    config = plates.PlatesConfig()


    # Override the training configurations with a few
    # changes for inferencing.
    class InferenceConfig(config.__class__):
        # Run detection on one image at a time
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        DETECTION_MIN_CONFIDENCE = 0.9


    config = InferenceConfig()
    config.display()

    # Device to load the neural network on.
    # Useful if you're training a model on the same
    # machine, in which case use CPU and leave the
    # GPU for training.
    DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0

    # Inspect the model in training or inference modes
    # values: 'inference' or 'training'
    # TODO: code for 'training' test mode not ready yet
    TEST_MODE = "inference"

    # Load validation dataset
    dataset = plates.PlateDataset()
    if args.dataset == 'test':
        dataset.load_plates(args.dataset, args.test_image_path)
    else:
        dataset.load_plates(args.dataset)

    # Must call before using the dataset
    dataset.prepare()

    print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))

    # Create model in inference mode
    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

    # Or, load the last model you trained
    # weights_path = model.find_last()
    weights_path = PLATES_WEIGHTS_PATH

    # Load weights
    print("Loading weights ", weights_path)
    model.load_weights(weights_path, by_name=True)

    for image_id in dataset.image_ids:

        path = os.path.basename(dataset.source_image_link(image_id))[:-4]
        print('Processed: ', path)
        if args.dataset == "test":
            image = dataset.load_image(image_id)
            original_shape = image.shape
            image, window, scale, padding, crop = utils.resize_image(
                image,
                min_dim=config.IMAGE_MIN_DIM,
                min_scale=config.IMAGE_MIN_SCALE,
                max_dim=config.IMAGE_MAX_DIM,
                mode=config.IMAGE_RESIZE_MODE)

            # Run object detection
            h, w = np.int32(original_shape[:2])
            y_reshape = window[0]
        else:
            image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset, config, image_id,
                                                                                      use_mini_mask=False)
            h, w = np.int32(image_meta[1:3])
            scale = image_meta[11]
            y_reshape = image_meta[7]

        results = model.detect([image], verbose=0)

        r = results[0]

        points = get_corner_points(r)

        line = ''

        if points is not None:
            plate = fix_perspective(image, points)

            points[:, 1] = points[:, 1] - y_reshape

            points = np.int32([np.around(points / scale, 0)])

            list_points = points[0]

            xtopl = list_points[0][0] / w
            xtopr = list_points[1][0] / w
            xbottomr = list_points[2][0] / w
            xbottoml = list_points[3][0] / w
            ytopl = list_points[0][1] / h
            ytopr = list_points[1][1] / h
            ybottomr = list_points[2][1] / h
            ybottoml = list_points[3][1] / h

            line = '4,{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},,\n'.format(xtopl, xtopr, xbottomr, xbottoml,
                                                                                          ytopl,
                                                                                          ytopr, ybottomr, ybottoml)
            cv2.imwrite(plates_path + '/' + path + ".jpg", plate)


        with open(results_path + '/' + path + ".txt", 'w') as f:
            f.write(line)