import os
import numpy as np
import skimage.io
from mrcnn import utils


class FassegDataset(utils.Dataset):
    """
    Load prepared Fasseg Dataset samples
    """
    def __init__(self, dataset_dir, masks_dir, labels):
        super(FassegDataset, self).__init__(self)
        self.dataset_dir = dataset_dir
        self.masks_dir = masks_dir
        self.labels = labels

        self.images_paths = []
        self.masks_paths = []
        self.masks = dict()

        # Background added by default at index 0
        for index, label in enumerate(self.labels[1:]):
            self.add_class(label, index + 1, label)

    def load_images(self):

        exts = ["jpg", "png", "bmp"]

        for root, subdirs, files in os.walk(self.dataset_dir):
            for file in files:
                for ext in exts:
                    if file.endswith("." + ext):
                        self.images_paths.append(os.path.join(root, file))

        for image_path in self.images_paths:

            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "face",
                image_id=os.path.basename(image_path)[:-4],
                path=image_path,
                width=width,
                height=height)

    def load_masks(self):
        """
        Load masks in memory retrieving from numpy arrays serialized
        """
        for root, subdirs, files in os.walk(self.masks_dir):
            for file in files:
                if file.endswith(".npy"):
                    self.masks_paths.append(os.path.join(root, file))

        for mask_file in self.masks_paths:
            # Load mask, same size of image with int class_ids for each pixel (no overlapping)
            # We need to convert it to boolean map of shape [width, height, class_ids]
            # and return the class_ids present in each level (typically all the labels are present)
            mask_loaded = np.load(mask_file)
            image_id = os.path.basename(mask_file[:-4])
            values, indices = np.unique(mask_loaded, return_inverse=True)
            processed_mask = np.zeros(shape=(mask_loaded.shape[0], mask_loaded.shape[1], len(values)), dtype=np.bool)

            for index, value in enumerate(values):

                # Indices are flattened, so we have to flatten also the mask and then reshape
                boolean_map = (mask_loaded.flatten() == value)
                boolean_map = boolean_map.reshape(mask_loaded.shape)
                processed_mask[:, :, index] = boolean_map

            self.masks[image_id] = (processed_mask, np.array(values, dtype=np.int32))

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "face":
            return info["face"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Retrieve masks for an image.
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a face dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "face":
            return super(self.__class__, self).load_mask(image_id)

        # Retrieve mask
        mask, class_ids = self.masks[image_info["id"]]

        # Return mask, and array of class IDs of each instance
        return mask, class_ids
