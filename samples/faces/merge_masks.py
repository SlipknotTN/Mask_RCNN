import argparse
import csv
import os

from glob import glob

import cv2
import numpy as np

"""
Merge single class masks created with extract_masks.py to create BW ground truth compatible with SPADE,
same format of ADE20K dataset.
"""

def do_parsing():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--gt_dir", required=True, type=str, help="GT directory")
    parser.add_argument("--output_dir", required=True, type=str, help="Output directory")
    parser.add_argument("--output_format", required=False, type=str, default="jpg", help="Mask images output extension")
    parser.add_argument("--classes_csv", required=True, type=str, help="CSV file with class-colors mapping")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = do_parsing()
    print(args)

    files = sorted(glob(args.gt_dir + "/*.png"))
    print(files)

    os.makedirs(args.output_dir, exist_ok=True)

    labels = []

    with open(args.classes_csv) as csv_file:
        reader = csv.reader(csv_file, delimiter=";")
        next(reader)
        for row in reader:
            labels.append(row[0])

    print(labels)

    basename = None
    merged_image = None

    for file in files:

        filename = os.path.basename(file)
        second_underscore_idx = filename[filename.find("_")+1:].find("_") + filename.find("_")+1
        label_name = filename[second_underscore_idx + 1 : filename.rfind("_")]
        print(label_name)
        original_image_name = filename[:second_underscore_idx]
        print(original_image_name)
        instance_image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

        if original_image_name != basename:
            # Save old image
            if basename:
                cv2.imwrite(os.path.join(args.output_dir, basename + "." + args.output_format), merged_image)

            # Load new image
            basename = original_image_name
            height, width = instance_image.shape[:2]
            print(height, width)

            # Create np zeros image
            merged_image = np.zeros(shape=(height, width), dtype=np.uint8)

        # Add value = label_idx where label_img == 255
        merged_image[instance_image == 255] = labels.index(label_name)
