"""
Konfiguration template file for image classification task
"""

# from lib.utils import get_data_augmentation_transforms
from palladio.utils import get_data_augmentation_transforms

from torch.utils.data.dataloader import DataLoader

# from torch.utils.data import WeightedRandomSampler, RandomSampler
from torch.utils.data import RandomSampler

from torch.nn import BCEWithLogitsLoss

from sklearn.metrics import precision_score, recall_score, f1_score

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
# XXX Set training dataset here
train_set = None

# Input transformation

# For validation have data augmentation level set to 0 (NO DA)
val_transforms = get_data_augmentation_transforms(
    0, INPUT_NORMALIZATION)

# XXX Set validation dataset here
val_set = None

# class_map_dict = {
    # 0: "T-shirt/top",
    # 1: "Trouser",
    # 2: "Pullover",
    # 3: "Dress",
    # 4: "Coat",
    # 5: "Sandal",
    # 6: "Shirt",
    # 7: "Sneaker",
    # 8: "Bag",
    # 9: "Ankle boot",
# }

# Misc

# Specify loss
criterion = BCEWithLogitsLoss()


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
    predicted = (outputs > 0.5).int()

    # total = targets.size(0)
    # acc = 100.*correct/total

    metrics_functions = {
        'precision': precision_score,
        'recall': recall_score,
        'f1': f1_score,
    }

    metrics = dict()

    for m, mf in metrics_functions.items():
        for avg in ['micro', 'macro']:
            metrics[f'{m}_{avg}'] = mf(targets, predicted, average=avg)

    return metrics


def print_batch_log(outputs, targets, loss, logger, batch_idx,
                    n_batches, print_every):

    """
    Outputs are logits, targets
    """
    STATUS_MSG = "Batches done: {}/{} | Loss: {:04f} | F1 (micro): {:04f} | F1 (macro): {:04f}"  # noqa

    metrics = build_metrics(outputs, targets)

    if (batch_idx + 1) % print_every == 0:
        logger.info(STATUS_MSG.format(
            batch_idx+1,
            n_batches,
            loss/(batch_idx+1),
            metrics["f1_micro"],
            metrics["f1_macro"]))


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
num_classes = None
