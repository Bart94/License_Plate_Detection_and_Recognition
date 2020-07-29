import argparse
import os
import sys
import cv2

import matplotlib.pyplot as plt
import tensorflow as tf

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
import mrcnn.model as modellib
from mrcnn import visualize

import characters


def split(arr, cond):
    return [arr[cond], arr[~cond]]


if __name__ == '__main__':

    # Directory to save logs and trained model
    MODEL_DIR = "model_data/weights_box/"

    # Path to Characters trained weights
    CHARACTERS_WEIGHTS_PATH = MODEL_DIR + "mask_rcnn_characters_0018.h5"  # TODO: update this path

    parser = argparse.ArgumentParser(description='MASK R-CNN plate detector')
    parser.add_argument('--test_image_path', required=False,
                        metavar="/path/to/test_images",
                        help="Path to test images")
    parser.add_argument("--annotated_path", required=False, type=str, nargs='?',
                        help="Path della cartella in cui salvare le targhe annotate.")
    parser.add_argument("--result_file", required=True, type=str, nargs='?',
                        help="File in cui salvare i risultati.")
    parser.add_argument("--dataset", required=True, type=str, nargs='?',
                        help="train - test - val")
    args = parser.parse_args()

    result_file = args.result_file
    annotated_path = args.annotated_path

    if annotated_path is not None and not os.path.exists(annotated_path):
        os.makedirs(annotated_path)

    config = characters.CharactersConfig()


    # Override the training configurations with a few
    # changes for inferencing.
    class InferenceConfig(config.__class__):
        # Run detection on one image at a time
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        DETECTION_MIN_CONFIDENCE = 0.7


    config = InferenceConfig()
    config.display()

    # Device to load the neural network on.
    # Useful if you're training a model on the same
    # machine, in which case use CPU and leave the
    # GPU for training.
    DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0

    # Inspect the model in training or inference modes
    # values: 'inference' or 'training'
    TEST_MODE = "inference"

    # Load validation dataset
    dataset = characters.CharacterDataset()
    if args.dataset == 'test':
        dataset.load_characters(args.dataset, args.test_image_path)
    else:
        dataset.load_characters(args.dataset)

    # Must call before using the dataset
    dataset.prepare()

    print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))

    # Create model in inference mode
    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

    # Or, load the last model you trained
    weights_path = CHARACTERS_WEIGHTS_PATH

    # Load weights
    print("Loading weights ", weights_path)
    model.load_weights(weights_path, by_name=True)

    with open('model_data/character_classes.txt') as f:
        class_list = f.read().splitlines()

    plate = ''

    for image_id in dataset.image_ids:

        path = os.path.basename(dataset.source_image_link(image_id))[:-4]

        image = dataset.load_image(image_id)
        results = model.detect([image], verbose=0)

        print('Processed: ', path)

        r = results[0]
        h, w = image.shape[:2]

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if r['rois'].size:
            print(len(r['scores']), r['scores'])
            min_y = r['rois'][:, 0].min()
            plate += path + '.jpg/'
            sort_y = split(r['rois'], r['rois'][:, 0] < (min_y + 0.3 * h))
            sort_class = split(r['class_ids'], r['rois'][:, 0] < (min_y + 0.3 * h))
            tmp = ''
            for i in range(len(sort_y)):
                ord_x = sorted(range(len(sort_y[i])), key=lambda k: sort_y[i][k][1])
                ord_class = sorted(range(len(sort_y[i])), key=lambda k: sort_y[i][k][1])
                for elem in ord_x:
                    curr_char = class_list[sort_class[i][elem] - 1]
                    plate += curr_char
                    tmp += curr_char
                    if annotated_path is not None:
                        y1, x1, y2, x2 = sort_y[i][elem]
                        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 0), 1)
                        cv2.putText(image, curr_char, (x1 - int(0.03 * w), y2), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                    (0, 255, 255), 1)

            if annotated_path is not None:
                if w >= 1.7 * h:
                    image = cv2.resize(image, (600, 150))
                else:
                    image = cv2.resize(image, (300, 300))
                cv2.imwrite(annotated_path + '/' + path + '.jpg', image)
            plate += '\n'

    with open(result_file, 'w') as f:
        f.write(plate)
