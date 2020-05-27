"""
Konfiguration template file for image classification task
"""

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

EXPERIMENT_NAME = "FashionMNIST"

# Dataset

# Data augmentation/transformation
DATA_AUGMENTATION_LEVEL = 0
INPUT_NORMALIZATION = None

# Training hyperparameters
# BATCH_SIZE = 32
# BATCH_SIZE = 64
BATCH_SIZE = 256

# Load datasets
train_transforms = get_data_augmentation_transforms(
    DATA_AUGMENTATION_LEVEL, INPUT_NORMALIZATION)

# Use dataset FashionMNIST
# https://github.com/zalandoresearch/fashion-mnist
train_set = FashionMNIST(
    '.', train=True, transform=train_transforms, download=True)

# Input transformation

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

# Misc

# Specify loss
criterion = CrossEntropyLoss()


def print_batch_log(outputs, targets, loss, logger, batch_idx,
                    n_batches, print_every):

    STATUS_MSG = "Batches done: {}/{} | Loss: {:04f} | Accuracy: {:04f}"

    _, predicted = outputs.max(1)

    total = targets.size(0)

    correct = predicted.eq(targets).sum().item()

    # Compute accuracy
    acc = 100.*correct/total

    if (batch_idx + 1) % print_every == 0:
        logger.info(STATUS_MSG.format(
            batch_idx+1,
            n_batches,
            loss/(batch_idx+1),
            acc))


def build_metrics(outputs, targets):
    """
    Compute stats based on output and target. Returns a dictionary containing
    all metrics that should be logged.

    Parameters
    ----------

    outputs : torch.Tensor
        The result of a forward pass of the net

    target :
        The target

    Return
    ------

    dict : The aggregated metrics
    """

    # Classification problem: extract predicted labels
    _, predicted = outputs.max(1)

    total = targets.size(0)

    # Compute accuracy
    correct = predicted.eq(targets).sum().item()

    acc = 100.*correct/total

    metrics = dict()
    metrics['accuracy'] = acc

    return metrics


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
