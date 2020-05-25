# Experiment specific imports
# from lib.dataset.peltarion import Peltarion
# import os

from torchvision.datasets import FashionMNIST

# from lib.utils import get_data_augmentation_transforms
from palladio.utils import get_data_augmentation_transforms

from torch.utils.data.dataloader import DataLoader

# from torch.utils.data import WeightedRandomSampler, RandomSampler
from torch.utils.data import RandomSampler

from torch.nn import CrossEntropyLoss

########################
####### Settings ####### #noqa
########################

# Dataset

# Data augmentation/transformation
DATA_AUGMENTATION_LEVEL = 0
INPUT_NORMALIZATION = None

# Training hyperparameters
BATCH_SIZE = 32

# Load datasets
train_transforms = get_data_augmentation_transforms(
    DATA_AUGMENTATION_LEVEL, INPUT_NORMALIZATION)

# Use dataset FashionMNIST
# https://github.com/zalandoresearch/fashion-mnist
train_set = FashionMNIST(
    '.', train=True, transform=train_transforms, download=True)

# For validation have data augmentation level set to 0 (NO DA)
val_transforms = get_data_augmentation_transforms(
    0, INPUT_NORMALIZATION)

val_set = FashionMNIST(
    '.', train=False, transform=val_transforms, download=True)

class_map_dict = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot",
}

# Specify loss
criterion = CrossEntropyLoss()

###################
####### END ####### #noqa
###################

# weights = train_set.make_weights_for_balanced_classes()

# N_samples_per_epoch = len(weights)
# This one is sort of eyeballed based on a few assumptions
# TODO explain
# N_samples_per_epoch = 50 * 7 * args.batch_size
# N_samples_per_epoch = 1000 * BATCH_SIZE

# For finding appropriate lr
# N_samples_per_epoch = 50 * 7
# train_sampler = WeightedRandomSampler(weights, N_samples_per_epoch)

train_sampler = RandomSampler(train_set)
val_sampler = RandomSampler(val_set)

# Use custom sampler for train_loader
train_loader = DataLoader(train_set,
                          batch_size=BATCH_SIZE,
                          sampler=train_sampler,
                          num_workers=6)

val_loader = DataLoader(val_set,
                        batch_size=BATCH_SIZE,
                        sampler=val_sampler,
                        num_workers=6)

# Specify number of classes
num_classes = len(class_map_dict)
