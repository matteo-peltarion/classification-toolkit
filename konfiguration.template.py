# Experiment specific imports
# from lib.dataset.peltarion import Peltarion
import os

from torchvision.datasets import ImageFolder

from lib.utils import get_data_augmentation_transforms

from torch.utils.data.dataloader import DataLoader

from torch.utils.data import WeightedRandomSampler, RandomSampler

from PIL import Image

########################
####### Settings ####### #noqa
########################

# Dataset
DATA_DIR = "/home/matteo/project/IPSOS-er/data/fer2013/fer_7"

INPUT_FEATURES = ["images"]

# Data augmentation/transformation
DATA_AUGMENTATION_LEVEL = 0
INPUT_NORMALIZATION = None

# Training hyperparameters
BATCH_SIZE = 32

# Load datasets
train_transforms = get_data_augmentation_transforms(
    DATA_AUGMENTATION_LEVEL, INPUT_NORMALIZATION)

# img_loader = lambda path: with open(path, 'rb') as f:
        # img = Image.open(f)

# This is done to prevent forcing load of a RGB image
img_loader = lambda path: Image.open(path)

###################
####### END ####### #noqa
###################

# TODO dataset specific, move somewhere else
class_map_dict = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral"
}

# TODO dataset specific, move somewhere else
# train_set = Peltarion(
    # DATA_DIR, 'train',
    # INPUT_FEATURES, target_column='emotion',
    # class_map_dict=class_map_dict,
    # transforms=train_transforms)

train_set = ImageFolder(
    os.path.join(DATA_DIR, 'fer_train'),
    transform=train_transforms,
    loader=img_loader)

train_set.class_map_dict = class_map_dict

# print(dir(train_set))
# print(train_set.classes)

# For validation have data augmentation level set to 0 (NO DA)
val_transforms = get_data_augmentation_transforms(
    0, INPUT_NORMALIZATION)

# val_set = Peltarion(
    # DATA_DIR, 'test',
    # INPUT_FEATURES, target_column='emotion',
    # class_map_dict=class_map_dict,
    # transforms=val_transforms)
val_set = ImageFolder(
    os.path.join(DATA_DIR, 'fer_val'),
    # transform=val_transforms)
    transform=train_transforms,
    loader=img_loader)

val_set.class_map_dict = class_map_dict

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
num_classes = 7
