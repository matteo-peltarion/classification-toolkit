"""
Konfiguration template file for image classification task
"""

from palladio.utils import get_data_augmentation_transforms

from torch.utils.data import RandomSampler

from palladio.networks.utils import get_network as pd_get_network

from torch.nn import BCEWithLogitsLoss

from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score)

########################
####### Settings ####### #noqa
########################

EXPERIMENT_NAME = "MLC"

# Dataset

# Data augmentation/transformation
DATA_AUGMENTATION_LEVEL = 0
INPUT_NORMALIZATION = None

# Load datasets
train_transforms = get_data_augmentation_transforms(
    DATA_AUGMENTATION_LEVEL, INPUT_NORMALIZATION)

# XXX Set training dataset here
train_set = None

# Input transformation

# For validation have data augmentation level set to 0 (NO DA)
val_transforms = get_data_augmentation_transforms(
    0, INPUT_NORMALIZATION)

# XXX Set validation dataset here
val_set = None

# This variable should be a dictionary whose keys are the index of a class and
# the values are strings describing the class.
class_map_dict = None

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

    metrics_functions = {
        'precision': precision_score,
        'recall': recall_score,
        'f1': f1_score,
    }

    metrics = dict()

    for m, mf in metrics_functions.items():
        for avg in ['micro', 'macro']:
            metrics[f'{m}_{avg}'] = mf(targets, predicted, average=avg)

    if class_map_dict is not None:

        # Add accuracy to metrics
        metrics_functions['accuracy'] = accuracy_score

        # Per class metrics
        for i, c in class_map_dict.items():

            # Sanitize name of the class
            cc = c.lower().replace(" ", "_")

            for m, mf in metrics_functions.items():
                metrics[f'{m}_{cc}'] = mf(
                    targets[:, i].cpu(), predicted[:, i].cpu())

    return metrics


def print_batch_log(outputs, targets, loss, logger, batch_idx,
                    n_batches, print_every, subset):

    """
    Outputs are logits, targets
    """
    STATUS_MSG = "[{}] Batches done: {}/{} | Loss: {:04f} | F1 (micro): {:04f} | F1 (macro): {:04f}"  # noqa

    metrics = build_metrics(outputs, targets)

    if (batch_idx + 1) % print_every == 0:
        logger.info(STATUS_MSG.format(
            subset,
            batch_idx+1,
            n_batches,
            loss/(batch_idx+1),
            metrics["f1_micro"],
            metrics["f1_macro"]))


###################
####### END ####### #noqa
###################

train_sampler = RandomSampler(train_set)
val_sampler = RandomSampler(val_set)

# Specify number of classes
num_classes = None


def get_network(network_name, use_pretrained):
    return pd_get_network(network_name, num_classes, use_pretrained)
