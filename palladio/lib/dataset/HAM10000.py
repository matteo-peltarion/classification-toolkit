"""Module to parse dataset."""

# Imports.
import os
from os.path import join
import glob

import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# import torch

from torch.utils.data import Dataset

# import torchvision

import torchvision.transforms as transforms


def read_image_paths(data_dir):
    """Read image paths from data directory.

    Args:
        data_dir (str): path to folder with images.

    Returns:
        image_paths (list): list of image paths.

    """
    image_extension_pattern = '*.jpg'
    image_paths = sorted((y for x in os.walk(data_dir) for y in
                          glob.glob(join(x[0], image_extension_pattern))))
    return image_paths


def get_image_paths_dict(data_dir):
    """Create and return dict that maps image IDs to image paths.

    Args:
        data_dir (str): path to folder with images

    Returns:
        image_paths_dict (dict): dict to map image IDs to image paths.

    """
    image_paths = read_image_paths(data_dir)
    image_paths_dict = {}
    for image_path in image_paths:
        image_id = image_path.split('/')[-1].split('.jpg')[0]
        image_paths_dict[image_id] = image_path

    return image_paths_dict


def read_meta_data(data_dir):
    """Read meta data file using Pandas.

    Returns:
        meta_data (pandas.core.frame.DataFrame): meta-data object.

    """
    meta_data = pd.read_csv(join(data_dir, 'HAM10000_metadata.csv'),
                            index_col='image_id')
    return meta_data


def load_image(image_path):
    """Load image as numpy array.

    Args:
        image_path (str): path to image.

    Returns:
        (numpy.ndarray): image as numpy array.

    """
    return np.array(Image.open(image_path))


def show_images(images, cols=1, titles=None):
    """Display multiple images arranged as a table.

    Args:
        images (list): list of images to display as numpy arrays.
        cols (int, optional): number of columns.
        titles (list, optional): list of title strings for each image.

    """
    assert((titles is None)or (len(images) == len(titles)))
    n_images = len(images)

    if titles is None:
        titles = ['Image (%d)' % i for i in range(1, n_images + 1)]

    fig = plt.figure()

    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)

    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()


def create_train_val_split(data_dir,
                           train_fraction, val_fraction,
                           random_state=123):
    """Split data into training and validation sets, based on given fractions.

    Args:
        train_fraction (float): fraction of data to use for training.
        val_fraction (float): fraction of data to use for training.
        random_state (int): seed for the random generator (for
                            reproducibility).

    Returns:
        (tuple): tuple with training image IDs and validation image IDs.

    """
    assert(train_fraction + val_fraction <= 1.0)

    # TODO move this somewhere else
    LABEL_COLUMN = 'dx'
    # IMAGE_ID_COLUMN = 'image_id'

    # TODO: Implement a proper training/validation split
    meta_data = read_meta_data(data_dir)

    train_ids = list()
    val_ids = list()

    for gg in meta_data.groupby(LABEL_COLUMN):
        # category = gg[0]
        group_df = gg[1]

        n = len(group_df)

        # Retrieve number of
        n_tr = int(n*train_fraction)

        # To get as much data as possible
        if (train_fraction + val_fraction == 1.0):
            n_vd = n - n_tr
        else:
            n_vd = int(n*val_fraction)

        # This is just used for shuffling, returns the whole group because of
        # frac=1.0
        group_ids = group_df.sample(
            frac=1.0, random_state=random_state).index.to_list()

        # Get ids for training and validation set
        group_ids_train = group_ids[:n_tr]
        group_ids_val = group_ids[n_tr:n_tr+n_vd]

        # Add ids to ids list
        train_ids.extend(group_ids_train)
        val_ids.extend(group_ids_val)

    return train_ids, val_ids


class HAM10000(Dataset):
    """HAM10000 dataset.

    Attributes:
        sampling_list (list): list of image IDs to use.
        image_paths_dict (dict): dict to map image IDs to image paths.
        meta_data (pandas.core.frame.DataFrame): meta data object.
        class_map_dict (dict): dict to map label strings to label indices.
        transforms ():

    """

    def __init__(self, data_dir, sampling_list,
                 transforms=transforms.ToTensor()):
        """Constructor.

        Args:
            data_dir (str): path to images and metadata file
            sampling_list (list): list of image IDs to use.

        """
        self.data_dir = data_dir
        self.sampling_list = sampling_list
        self.image_paths_dict = get_image_paths_dict(self.data_dir)
        self.meta_data = read_meta_data(self.data_dir)
        self.class_map_dict = self.get_class_map_dict()

        self.transforms = transforms


    def get_labels(self):
        """Get labels of dataset and return them as list.

        Returns:
            (list): list of all labels.

        """
        labels = [self.meta_data.loc[image_id]['dx']
                  for image_id in self.sampling_list]

        return labels

    def get_num_classes(self):
        """Get number of classes.

        Returns:
            (int): number of classes.

        """
        return len(self.class_map_dict)

    def get_class_map_dict(self):
        """Get dict to map label strings to label indices.

        Returns:
            class_map_dict (dict): dict to map label strings to label indices.

        """
        classes_list = list(
            self.meta_data.groupby('dx')['lesion_id'].nunique().keys())

        classes_list = sorted(classes_list)
        class_map_dict = {}
        for i, cls in enumerate(classes_list):
            class_map_dict[cls] = i

        return class_map_dict

    def __len__(self):
        """Get size of dataset.

        Returns:
            (int): size of dataset, i.e. number of samples.

        """
        return len(self.sampling_list)

    def __getitem__(self, index):
        """Get item.

        Args:
            index (int): index.

        Returns:
            (tuple): tuple with image and label.

        """
        image_id = self.sampling_list[index]
        img = Image.open(self.image_paths_dict.get(image_id))
        assert(image_id in self.meta_data.index)
        label = self.class_map_dict[self.meta_data.loc[image_id]['dx']]

        img = self.transforms(img)

        return img, label

    def make_weights_for_balanced_classes(self):
        """Function used to return weights for WeightedRandomSampler

        Inspired by:
            https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/3
        """

        count = [0] * self.get_num_classes()

        # label = self.class_map_dict[self.meta_data.loc[image_id]['dx']]
        labels = [self.class_map_dict[l] for l in self.get_labels()]

        # print(labels)

        # Count how many instances there are for each class
        for l in labels:
            count[l] += 1

        weight_per_class = [0.] * self.get_num_classes()

        N = float(sum(count))

        # Assign a weight which is inversely proportional to class frequency
        for i in range(self.get_num_classes()):
            weight_per_class[i] = N/float(count[i])

        # print("Weights per class:")
        # print(weight_per_class)

        # Save results for debugging purposes
        self._weight_per_class = weight_per_class

        # Now assign a weight to each data point
        weight = [0] * len(labels)

        for idx, val in enumerate(labels):
            weight[idx] = weight_per_class[val]

        return weight
