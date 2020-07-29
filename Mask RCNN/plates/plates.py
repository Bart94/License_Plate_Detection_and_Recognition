"""
Mask R-CNN

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 plates.py --weights=coco

    # Resume training a model that you had trained earlier
    python3 plates.py --weights=last

    # Train a new model starting from ImageNet weights
    python3 plates.py --weights=imagenet

"""
import json
import os
import sys

import imgaug.augmenters as iaa
import numpy as np
import skimage.draw
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from glob import glob


# Root directory of the project
ROOT_DIR = os.path.abspath("../")
DATASETS_DIR = os.path.abspath("../../Datasets/")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

from PIL import ImageFile

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


############################################################
#  Configurations
############################################################


class PlatesConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    NAME = "plate"

    IMAGES_PER_GPU = 1

    NUM_CLASSES = 1 + 1

    DETECTION_MIN_CONFIDENCE = 0.9

    STEPS_PER_EPOCH = 1000

    MAX_GT_INSTANCES = 5


############################################################
#  Dataset
############################################################

class PlateDataset(utils.Dataset):

    def load_plates(self, subset, test_image_path=None):
        """Load a subset of the Plates dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train, val or test
        """
        # Add classes. We have only one class to add.
        self.add_class("plate", 1, "plate")

        # Train, validation or test dataset?
        assert subset in ["train", "val", "test"]

        if subset == "test":
            for path in glob(os.path.join(test_image_path, '*.jpg')):
                path = os.path.basename(path)
                image_path = test_image_path + "/" + path

                print('Dataset: ', path)
                self.add_image("characters", image_id=path, path=image_path)
        else:
            # Load annotations
            annotations = json.load(open("model_data/data_" + subset + ".json"))
            annotations = list(annotations.values())  # don't need the dict keys

            annotations = [a for a in annotations if a['regions']]

            # Add images
            for a in annotations:
                # Get the x, y coordinaets of points of the polygons that make up
                # the outline of each object instance. These are stores in the
                # shape_attributes (see json format above)
                if type(a['regions']) is dict:
                    polygons = [r['shape_attributes'] for r in a['regions'].values()]
                else:
                    polygons = [r['shape_attributes'] for r in a['regions']]

                # load_mask() needs the image size to convert polygons to masks.
                ImageFile.LOAD_TRUNCATED_IMAGES = True
                image_path = DATASETS_DIR + "/" + a['filename']
                image = skimage.io.imread(image_path)
                height, width = image.shape[:2]

                self.add_image(
                    "plate",
                    image_id=a['filename'],  # use file name as a unique image id
                    path=image_path,
                    width=width, height=height,
                    polygons=polygons)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a plate dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "plate":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            rr[rr > mask.shape[0]-1] = mask.shape[0]-1
            cc[cc > mask.shape[1]-1] = mask.shape[1]-1
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "plate":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = PlateDataset()
    dataset_train.load_plates("train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = PlateDataset()
    dataset_val.load_plates("val")
    dataset_val.prepare()

    #Augmentation
    aug = iaa.OneOf([
        iaa.GaussianBlur(sigma=(0, 1.0)),
        iaa.MotionBlur(),
        iaa.RemoveSaturation((0.0, 0.5)),
        iaa.GammaContrast(),
        iaa.Rotate(rotate=(-45, 45)),
        iaa.PerspectiveTransform(scale=(0.01, 0.15)),
        iaa.SaltAndPepper(),
        iaa.JpegCompression(compression=(0, 75)),
        iaa.imgcorruptlike.Spatter(severity=(1, 4)),
        iaa.imgcorruptlike.DefocusBlur(severity=1)
    ])

    custom_callbacks = [ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1),
                        EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)]

    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=100,
                layers='all',
                augmentation=aug,
                custom_callbacks=custom_callbacks)


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect plates.')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    args = parser.parse_args()

    # Validate arguments
    print("Weights: ", args.weights)
    print("Logs: ", args.logs)

    # Configurations
    config = PlatesConfig()
    config.display()

    # Create model
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    train(model)