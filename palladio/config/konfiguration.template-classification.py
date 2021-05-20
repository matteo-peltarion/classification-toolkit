"""
Konfiguration template file for image classification task
"""

from torchvision.datasets import FashionMNIST

from palladio.utils import get_data_augmentation_transforms

from palladio.networks.utils import get_network as pd_get_network

from torch.utils.data import RandomSampler

from torch.nn import CrossEntropyLoss

########################
####### Settings ####### #noqa
########################

# Sacred connection parameters
SACRED_DB_HOST = 'localhost'
SACRED_DB_PORT = 27017
SACRED_DB_USERNAME = 'sample'
SACRED_DB_PASSWORD = 'password'
SACRED_DB_NAME = 'db'

EXPERIMENT_NAME = "FashionMNIST"

# Dataset

# Data augmentation/transformation
DATA_AUGMENTATION_LEVEL = 0
INPUT_NORMALIZATION = None

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

scheduler_kwargs = dict(
    mode='min', factor=0.5, patience=5, min_lr=1e-8
)


def print_batch_log(outputs, targets, loss, logger, batch_idx,
                    n_batches, print_every, subset):

    STATUS_MSG = "[{}] Batches done: {}/{} | Loss: {:04f} | Accuracy: {:04f}"

    _, predicted = outputs.max(1)

    total = targets.size(0)

    correct = predicted.eq(targets).sum().item()

    # Compute accuracy
    acc = 100.*correct/total

    if (batch_idx + 1) % print_every == 0:
        logger.info(STATUS_MSG.format(
            subset,
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


def is_best_model(new_metrics, current_best):
    """
    Returns True if the reference metric for the current model is better than
    the previous best.
    """

    is_best = new_metrics['accuracy'] > current_best
    if is_best:
        current_best = new_metrics['accuracy']

    return is_best, current_best

###################
####### END ####### #noqa
###################


train_sampler = RandomSampler(train_set)
val_sampler = RandomSampler(val_set)

# Specify number of classes
num_classes = len(class_map_dict)


def get_network(network_name, use_pretrained):
    return pd_get_network(
        network_name, num_classes, use_pretrained, n_input_channels=1)
