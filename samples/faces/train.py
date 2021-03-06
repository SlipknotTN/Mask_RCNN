import os
import sys
import argparse
import csv

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from samples.faces.FassegDataset import FassegDataset
from samples.faces.FaceConfig import FaceConfig
from mrcnn import utils
import mrcnn.model as modellib

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")


def load_weights(model, init_with):
    if init_with == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    elif init_with == "coco":
        # Load weights trained on MS COCO, but skip layers that
        # are different due to the different number of classes
        model.load_weights(COCO_MODEL_PATH, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                    "mrcnn_bbox", "mrcnn_mask"])
    elif init_with == "last":
        # Load the last model you trained and continue training
        model.load_weights(model.find_last(), by_name=True)


def do_parsing():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--images_train_dir", required=True, type=str, help="Images train directory")
    parser.add_argument("--masks_train_dir", required=True, type=str, help="Masks train directory")
    parser.add_argument("--images_val_dir", required=True, type=str, help="Images val directory")
    parser.add_argument("--masks_val_dir", required=True, type=str, help="Masks val directory")
    parser.add_argument("--output_dir", required=False, type=str, default="./model",
                        help="Mask images output extension")
    parser.add_argument("--init_weights", required=False, type=str, default="coco",
                        help="Backbone model starting weights")
    parser.add_argument("--classes_csv", required=True, type=str, help="CSV file with class-colors mapping")
    args = parser.parse_args()
    return args


def main():

    args = do_parsing()
    print(args)

    # Download COCO trained weights from Releases if needed
    if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)

    # Load config
    config = FaceConfig()
    config.display()

    # Read labels
    labels = []

    with open(args.classes_csv, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        # Skip header
        next(reader)
        for row in reader:
            labels.append(row[0])

    # Training dataset
    dataset_train = FassegDataset(dataset_dir=args.images_train_dir, masks_dir=args.masks_train_dir, labels=labels)
    dataset_train.load_images()
    dataset_train.load_masks()
    dataset_train.prepare()

    # Validation dataset
    dataset_val = FassegDataset(dataset_dir=args.images_val_dir, masks_dir=args.masks_val_dir, labels=labels)
    dataset_val.load_images()
    dataset_val.load_masks()
    dataset_val.prepare()

    # Load model pretrained weight

    # Create model in training mode
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=args.output_dir)
    load_weights(model, args.init_weights)

    # Train model

    # Train the head branches
    # Passing layers="heads" freezes all layers except the head
    # layers. You can also pass a regular expression to select
    # which layers to train by name pattern.
    print("Training head")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=3,
                layers='heads')

    # Fine tune all layers
    # Passing layers="all" trains all layers. You can also
    # pass a regular expression to select which layers to
    # train by name pattern.
    print("Fine tuning all")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=2,
                layers="all")

    print("Success")


if __name__ == "__main__":
    main()