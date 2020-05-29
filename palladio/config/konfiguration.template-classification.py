"""
Konfiguration template file for image classification task
"""

from torchvision.datasets import FashionMNIST

# from lib.utils import get_data_augmentation_transforms
from palladio.utils import get_data_augmentation_transforms

from palladio.networks.utils import get_network as pd_get_network

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

    # # TODO add this to parameter
    # if (epoch + 1) % 10 == 0:
        # # Display confusion matrix
        # cm = confusion_matrix(all_targets, all_predicted)

        # # Save confusion matrix
        # np.save(os.path.join(
            # exp_dir, "confusion_matrix_train_latest.npy"), cm)

        # # TODO check what happens if class_map_dict is not set (also for
        # # validation)

        # # Get detailed stats per class
        # stats_per_class = produce_per_class_stats(
            # all_targets, all_predicted, class_map_dict)

        # # Add scalars corresponding to these metrics to tensorboard
        # for score in ['precision_score', 'recall_score',
                      # 'roc_auc_score']:  # noqa
            # for k in class_map_dict:
                # # Add scalars to tb
                # writer.add_scalar(
                    # "{}_{}_train".format(k, score),
                    # stats_per_class[k][score],
                    # epoch)

        # cm_pretty = cm2df(cm, class_map_dict)

        # print(cm_pretty)

        # # Compute balanced accuracy
        # bal_acc = balanced_accuracy_score(all_targets, all_predicted)

        # writer.add_scalar("balanced_accuracy/train", bal_acc, epoch)

    # # Add accuracy on validation set to tb
    # writer.add_scalar("accuracy/train", acc, epoch)

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

# Specify number of classes
num_classes = len(class_map_dict)


def get_network(network_name, use_pretrained):
    return pd_get_network(network_name, num_classes, use_pretrained)
