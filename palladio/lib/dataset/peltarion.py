"""Module to parse dataset."""

# Imports.
import os
from os.path import join

import pandas as pd
# from PIL import Image
import numpy as np

# import torch

from torch.utils.data import Dataset

# import torchvision

import torchvision.transforms as transforms


def read_meta_data(data_dir):
    """Read meta data file using Pandas.

    Returns:
        meta_data (pandas.core.frame.DataFrame): meta-data object.

    """
    meta_data = pd.read_csv(join(data_dir, 'index.csv'))

    return meta_data


class Peltarion(Dataset):
    """Peltarion platform dataset.

    Created for instance using the pip `sidekick` tool.

    Attributes:
        sampling_list (list): list of image IDs to use.
        image_paths_dict (dict): dict to map image IDs to image paths.
        meta_data (pandas.core.frame.DataFrame): meta data object.
        class_map_dict (dict): dict to map label strings to label indices.
        transforms ():

    """

    def __init__(self, data_dir, subset,
                 input_features, target_column=None,
                 class_map_dict=None, cache_inputs=True,
                 transforms=transforms.ToTensor()):
        """Constructor.

        Parameters
        ----------

            data_dir : str
                path to images and metadata file

            input_features : List[str]
                The list of features to be used as input

            target_column : str
                The column used as target

            class_map_dict : dict
                The dictionary containing the mapping from target to category
                name.

            cache_inputs : bool
                Whether to cache input and keep them in memory or load them
                from disk every time. Default True.

            subset : str
                Either 'training' or 'test'

        """
        self.data_dir = data_dir
        self.subset = subset
        self.input_features = input_features
        self.target_column = target_column
        self.class_map_dict = class_map_dict
        self.cache_inputs = cache_inputs

        # Keep samples from the right subset only
        self.meta_data = read_meta_data(self.data_dir)
        self.meta_data = self.meta_data[self.meta_data['subset'] == subset]

        # Must recreate index
        self.meta_data = self.meta_data.reset_index()

        # Cache inputs
        self.cached_inputs = dict()

        self.transforms = transforms

    def get_labels(self):
        """Get labels of dataset and return them as list.

        Returns:
            (list): list of all labels.

        """

        labels = list(self.meta_data[self.target_column])

        return labels

    def get_num_classes(self):
        """Get number of classes.

        Returns:
            (int): number of classes.

        """
        return len(self.class_map_dict)

    def __len__(self):
        """Get size of dataset.

        Returns:
            (int): size of dataset, i.e. number of samples.

        """
        return len(self.meta_data)

    def __getitem__(self, index):
        """Get item.

        Args:
            index (int): index.

        Returns:
            (tuple): tuple with image and label.

        """

        if self.cache_inputs and index in self.cached_inputs:
            img = self.cached_inputs[index]
        else:

            channels = [
                np.load(os.path.join(
                    self.data_dir,
                    self.meta_data.loc[index][feature_name]))
                for feature_name in self.input_features]

            img = np.stack(channels, axis=-1)

            if self.cached_inputs:
                # Cache img
                self.cached_inputs[index] = img

        # Apply transforms
        img = self.transforms(img)

        label = self.meta_data.loc[index][self.target_column]

        return img, label

    def make_weights_for_balanced_classes(self):
        """Function used to return weights for WeightedRandomSampler

        Inspired by:
            https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/3
        """

        count = [0] * self.get_num_classes()

        # label = self.class_map_dict[self.meta_data.loc[image_id]['dx']]
        # labels = [self.class_map_dict[l] for l in self.get_labels()]

        labels = self.get_labels()

        # Count how many instances there are for each class
        for l in labels:
            count[l] += 1

        weight_per_class = [0.] * self.get_num_classes()

        N = float(sum(count))

        # Assign a weight which is inversely proportional to class frequency
        for i in range(self.get_num_classes()):
            weight_per_class[i] = N/float(count[i])

        # Save results for debugging purposes
        self._weight_per_class = weight_per_class

        # Now assign a weight to each data point
        weight = [0] * len(labels)

        for idx, val in enumerate(labels):
            weight[idx] = weight_per_class[val]

        return weight
