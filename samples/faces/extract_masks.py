import argparse
import glob
import os
import csv
import cv2
import numpy as np
from tqdm import tqdm
import scipy.spatial as spatial


def do_parsing():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--images_dir", required=True, type=str, help="Images directory")
    parser.add_argument("--output_dir", required=True, type=str, help="Output directory")
    parser.add_argument("--output_format", required=False, type=str, default="jpg", help="Mask images output extension")
    parser.add_argument("--classes_csv", required=True, type=str, help="CSV file with class-colors mapping")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    args = parser.parse_args()
    return args


def match_nearest_center(classes, centers):

    matches = []

    for label_index, class_instance in enumerate(classes):

        label = class_instance[0]

        min_distance = 10000.0
        min_index = -1

        for index, center in enumerate(centers):

            distance = spatial.distance.euclidean(center, np.array(classes[label_index][1]))
            if distance < min_distance:
                min_index = index
                min_distance = distance

        matches.append((label, min_index))

    return matches


def remap_clusters(clusters, matches):
    """
    Create remapped cluster assignments with desired indexes
    """
    clusters_remapped = np.zeros_like(clusters)
    for dest_index, match in enumerate(matches):
        src_index = match[1]
        boolean_map = (clusters == src_index)
        clusters_remapped[boolean_map] = dest_index
    return clusters_remapped


def main():
    # Load params
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
            colors = tuple([float(color) for color in colors])
            assert len(colors) == 3, "Wrong number of colors for " + row[0]
            classes.append((row[0], colors))

    os.makedirs(args.output_dir, exist_ok=False)

    subdirs = next(os.walk(args.images_dir))[1]

    exts = ["bmp", "jpg", "png"]

    for subdir in tqdm(subdirs, desc="subdir"):

        os.makedirs(os.path.join(args.output_dir, subdir), exist_ok=False)

        files = []

        for ext in exts:

            files += glob.glob(os.path.join(args.images_dir, subdir) + "/*." + ext)

            for file in tqdm(files, desc="image"):

                image = cv2.imread(file, cv2.IMREAD_COLOR)
                shape = image.shape

                if args.debug:

                    cv2.imshow("Image", image)
                    cv2.waitKey(0)

                Z = image.reshape((-1, 3))

                # convert to np.float32
                Z = np.float32(Z)

                # define criteria, number of clusters(K) and apply kmeans()
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                K = len(classes)
                ret, clusters, centers = cv2.kmeans(Z, K, None, criteria, 30, cv2.KMEANS_RANDOM_CENTERS)

                matches = match_nearest_center(classes, centers)

                # Export clusters for each pixel as numpy array
                # (but with labels index, not clusters ones), one for each image
                clusters_remapped = remap_clusters(clusters, matches)
                clusters_remapped = clusters_remapped.reshape((shape[0], shape[1]))

                np.save(os.path.join(args.output_dir, subdir, os.path.basename(file[:-4])) + ".npy", clusters_remapped)

                # Build black and white images for each label value (K)
                for k in range(len(matches)):

                    label = matches[k][0]
                    #print("Label: " + label)
                    label_index = matches[k][1]
                    #print("Index: " + str(label_index))
                    mask = [clusters == label_index][0]
                    mask = mask.reshape((shape[0], shape[1]))
                    mask = mask.astype('uint8') * 255

                    if args.debug:
                        cv2.imshow("BW", mask)
                        cv2.waitKey(0)

                    cv2.imwrite(os.path.join(args.output_dir, subdir,
                                             os.path.basename(file[:-4])) + "_" + label + "." + args.output_format,
                                mask)

    print("Success")


if __name__ == "__main__":
    main()
