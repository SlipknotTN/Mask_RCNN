import os
import sys
import argparse
import csv
import numpy as np

from tqdm import tqdm
import skimage.io

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from samples.faces.FaceConfig import FaceConfig
import mrcnn.model as modellib


def do_parsing():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--images_dir", required=True, type=str, help="Images directory")
    parser.add_argument("--model_path", required=False, type=str, help="H5 model file path")
    parser.add_argument("--output_dir", required=False, type=str, help="Output directory with predictions")
    parser.add_argument("--output_format", required=False, type=str, default="jpg", help="Mask images output extension")
    parser.add_argument("--classes_csv", required=True, type=str, help="CSV file with class-colors mapping")
    args = parser.parse_args()
    return args


def main():

    args = do_parsing()
    print(args)

    # Read classes
    classes = []

    with open(args.classes_csv, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        # Skip header
        next(reader)
        for row in reader:
            colors = row[1].split(",")
            colors = tuple([int(color) for color in colors])
            # Convert to RGB
            colors = colors[::-1]
            assert len(colors) == 3, "Wrong number of colors for " + row[0]
            classes.append((row[0], colors))

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
            colors = tuple([int(color) for color in colors])
            colors = colors[::-1]
            assert len(colors) == 3, "Wrong number of colors for " + row[0]
            classes.append((row[0], colors))

    # Load model and weights

    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=os.path.dirname(args.model_path), config=config)

    # Load weights trained
    model.load_weights(args.model_path, by_name=True)

    # Load a random image from the images folder
    exts = ["bmp", "jpg", "png"]

    os.makedirs(args.output_dir, exist_ok=True)

    images_paths = []

    for root, subdirs, files in os.walk(args.images_dir):
        for file in files:
            for ext in exts:
                if file.endswith("." + ext):
                    images_paths.append(os.path.join(root, file))
                    os.makedirs(os.path.join(args.output_dir, os.path.relpath(root, args.images_dir)), exist_ok=True)

    for file in tqdm(images_paths, desc="image"):

        # RGB
        image = skimage.io.imread(file)

        # Run detection
        results = model.detect([image], verbose=1)

        # Visualize results
        r = results[0]
        prediction = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.int32)

        # Apply background color
        prediction[::] = classes[0][1]

        for index, class_instance in enumerate(classes[1:]):
            class_color = np.array(class_instance[1])
            class_mask = r['masks'][:,:,index]
            prediction[class_mask] = class_color

        dest_path = os.path.join(args.output_dir, os.path.relpath(file, args.images_dir))
        dest_path = dest_path[:-3] + args.output_format
        skimage.io.imsave(dest_path, prediction)

    print("Success")


if __name__ == "__main__":
    main()