import os
import sys
import argparse
import csv

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from samples.faces.FaceConfig import FaceConfig
from mrcnn import utils
import mrcnn.model as modellib


def do_parsing():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--images_dir", required=True, type=str, help="Images directory")
    parser.add_argument("--model_path", required=False, type=str, help="H5 model file path")
    parser.add_argument("--output_dir", required=False, type=str, help="Output directory with predictions")
    parser.add_argument("--classes_csv", required=True, type=str, help="CSV file with class-colors mapping")
    args = parser.parse_args()
    return args


def main():

    args = do_parsing()
    print(args)

    # Load config
    class InferenceConfig(FaceConfig):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    config = InferenceConfig()
    config.display()

    # Read classes
    classes = []

    with open(args.classes_csv, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        # Skip header
        next(reader)
        for row in reader:
            colors = row[1].split(",")
            colors = tuple([float(color) for color in colors])
            assert len(colors) == 3, "Wrong number of colors for " + row[0]
            classes.append((row[0], colors))

    # Load model and weights

    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=os.path.dirname(args.model_path), config=config)

    # Load weights trained
    model.load_weights(args.model_path, by_name=True)

    # TODO: Adapt this: run on all test images, create output directories and save prediction images

    # Load a random image from the images folder
    file_names = next(os.walk(IMAGE_DIR))[2]
    image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))

    # Run detection
    results = model.detect([image], verbose=1)

    # Visualize results
    r = results[0]
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                class_names, r['scores'])

    print("Success")


if __name__ == "__main__":
    main()