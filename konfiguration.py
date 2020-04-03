# Experiment specific imports
from lib.dataset.peltarion import Peltarion

from lib.utils import get_data_augmentation_transforms

from torch.utils.data.dataloader import DataLoader

from torch.utils.data import WeightedRandomSampler, RandomSampler

########################
####### Settings ####### #noqa
########################

# Dataset
DATA_DIR = "/data"
INPUT_FEATURES = ["images"]

# Data augmentation/transformation
DATA_AUGMENTATION_LEVEL = 0
INPUT_NORMALIZATION = None

# Training hyperparameters
BATCH_SIZE = 16

# Load datasets
train_transforms = get_data_augmentation_transforms(
    DATA_AUGMENTATION_LEVEL, INPUT_NORMALIZATION)

###################
####### END ####### #noqa
###################

# TODO dataset specific, move somewhere else
class_map_dict = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral"
}

# TODO dataset specific, move somewhere else
train_set = Peltarion(
    DATA_DIR, 'train',
    INPUT_FEATURES, target_column='emotion',
    class_map_dict=class_map_dict,
    transforms=train_transforms)

# For validation have data augmentation level set to 0 (NO DA)
val_transforms = get_data_augmentation_transforms(
    0, INPUT_NORMALIZATION)

val_set = Peltarion(
    DATA_DIR, 'test',
    INPUT_FEATURES, target_column='emotion',
    class_map_dict=class_map_dict,
    transforms=val_transforms)

weights = train_set.make_weights_for_balanced_classes()

# N_samples_per_epoch = len(weights)
# This one is sort of eyeballed based on a few assumptions
# TODO explain
# N_samples_per_epoch = 50 * 7 * args.batch_size
N_samples_per_epoch = 1000 * BATCH_SIZE

# For finding appropriate lr
# N_samples_per_epoch = 50 * 7
train_sampler = WeightedRandomSampler(weights, N_samples_per_epoch)
# train_sampler = RandomSampler(train_set)
val_sampler = RandomSampler(val_set)

# Use custom sampler for train_loader
train_loader = DataLoader(train_set,
                          batch_size=BATCH_SIZE,
                          sampler=train_sampler,
                          num_workers=8)

val_loader = DataLoader(val_set,
                        batch_size=BATCH_SIZE,
                        sampler=val_sampler,
                        num_workers=8)
