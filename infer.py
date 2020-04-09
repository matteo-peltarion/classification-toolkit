#!/usr/bin/env python
"""
Runs a model in inference mode
"""

import os

import time

import datetime

import logging

import argparse

# from tqdm import tqdm

import matplotlib

matplotlib.use('agg')

# from sklearn.metrics import confusion_matrix, balanced_accuracy_score

# import numpy as np

# Torch stuff
# import torch.optim as optim
import torch.nn as nn

# from torch.utils.tensorboard import SummaryWriter

import torch
# import torch.backends.cudnn as cudnn

# Network options
from networks.SimpleCNN import SimpleCNN

from torchvision.models.vgg import vgg16
from torchvision.models.alexnet import alexnet
from torchvision.models.resnet import (
    resnet50, resnet34, resnet101, resnet152)

# from lib.utils import (
    # setup_logging, create_loss_plot, cm2df,
    # produce_per_class_stats)

from lib.utils import setup_logging

from PIL import Image

# Collect constants in separate file, C headers style.
# import constants

# Import experiment specific stuff
# from konfiguration import dataset, network
# from konfiguration import dataset
from konfiguration import (
    num_classes, val_transforms, class_map_dict)


def parse_args():

    # Parse args.
    parser = argparse.ArgumentParser(
        description='PyTorch classifier: inference mode')

    parser.add_argument(
        'test_data', help='Path to test data, file or folder.')

    # TODO move to konfiguration probably
    # parser.add_argument('--data-dir', help='Path to data', required=True)

    # parser.add_argument('--input-features', nargs='+', type=str,
                        # help='List of columns where inpute features are',
                        # metavar='FEATURE_NAME',
                        # required=True)

    parser.add_argument('--exp-name', default='baseline', type=str,
                        help='name of experiment')

    parser.add_argument('--log-level', default='INFO',
                        choices=['DEBUG', 'INFO'], help='log-level to use')

    parser.add_argument('--batch-size', default=4, type=int,
                        help='batch-size to use')

    # parser.add_argument('--class-weights', nargs='+', type=float,
                        # help='Weights for class (used for loss)')

    parser.add_argument('--network', default='SimpleCNN',
                        choices=[
                            'SimpleCNN', 'VGG16', 'Alexnet',
                            'resnet34', 'resnet50', 'resnet101', 'resnet152'],
                        help='network architecture')

    # parser.add_argument('--use-pretrained',
                        # action='store_true',
                        # help='Use a pretrained model.')  # noqa

    # parser.add_argument('--num-epochs', default=10, type=int,
                        # help='Number of training epochs')

    # parser.add_argument('--lr-finder',
                        # action='store_true',
                        # help='Exploratory run for finding best lr finder.')  # noqa

    parser.add_argument('--normalize-input',
                        action='store_true',
                        help='Normalize input using mean and variance computed on training set')  # noqa

    # parser.add_argument('--optimizer', default='Adam',
                        # choices=['Adam', 'SGD'],
                        # help='Optimizer.')

    # parser.add_argument('--resume', '-r',
                        # action='store_true', help='resume from checkpoint')

    args = parser.parse_args()

    # Must exist for reasons
    args.use_pretrained = False

    # print(dir(args))
    # print((args))

    return args


# Training.
# def train(net, train_loader, criterion, optimizer,
          # batch_size, device, epoch, logger, writer, exp_dir,
          # print_every=50):
    # """Performs training for one epoch

    # print_every : int
        # Print message only every N batches. Default 50.
    # """

    # logger.info('Epoch: {}'.format(epoch + 1))

    # net.train()

    # train_loss = 0
    # correct = 0
    # total = 0

    # # Needs to be changed because of the DL sampler
    # # n_batches = len(train_loader.dataset) // batch_size
    # n_batches = len(train_loader)

    # all_targets = np.array([], dtype=int)
    # all_predicted = np.array([], dtype=int)

    # for batch_idx, (inputs, targets) in enumerate(train_loader):

        # # Send tensors to the appropriate device
        # inputs, targets = inputs.to(device), targets.to(device)

        # # Set gradients to 0
        # optimizer.zero_grad()

        # # Forward pass
        # outputs = net(inputs)

        # # Compute loss
        # loss = criterion(outputs, targets)

        # # Backprop
        # loss.backward()
        # optimizer.step()

        # train_loss += loss.item()

        # _, predicted = outputs.max(1)

        # total += targets.size(0)

        # correct += predicted.eq(targets).sum().item()

        # # Compute accuracy
        # acc = 100.*correct/total

        # if (batch_idx + 1) % print_every == 0:
            # logger.info(constants.STATUS_MSG.format(
                # batch_idx+1,
                # n_batches,
                # train_loss/(batch_idx+1),
                # acc))

        # # TODO add this to parameter
        # if (epoch + 1) % 10 == 0:
            # # Save all for confusion matrix
            # all_targets = np.hstack(
                # (all_targets, targets.cpu().numpy().astype(int)))

            # all_predicted = np.hstack(
                # (all_predicted, predicted.cpu().numpy().astype(int)))

    # # TODO add this to parameter
    # if (epoch + 1) % 10 == 0:
        # # Display confusion matrix
        # cm = confusion_matrix(all_targets, all_predicted)

        # # Save confusion matrix
        # np.save(os.path.join(exp_dir, "confusion_matrix_train_latest.npy"), cm)

        # # TODO check what happens if class_map_dict is not set (also for
        # # validation)

        # # Get detailed stats per class
        # stats_per_class = produce_per_class_stats(
            # all_targets, all_predicted, train_loader.dataset.class_map_dict)

        # # Add scalars corresponding to these metrics to tensorboard
        # for score in ['precision_score', 'recall_score',
                      # 'roc_auc_score']:
            # for k in train_loader.dataset.class_map_dict:
                # # Add scalars to tb
                # writer.add_scalar(
                    # "{}_{}_train".format(k, score),
                    # stats_per_class[k][score],
                    # epoch)

        # cm_pretty = cm2df(cm, train_loader.dataset.class_map_dict)

        # print(cm_pretty)

        # # Compute balanced accuracy
        # bal_acc = balanced_accuracy_score(all_targets, all_predicted)

        # writer.add_scalar("balanced_accuracy/train", bal_acc, epoch)

    # # Add accuracy on validation set to tb
    # writer.add_scalar("accuracy/train", acc, epoch)

    # return train_loss/(batch_idx+1), acc


# def test(net, val_loader, criterion,
         # batch_size, device, epoch, logger, writer, exp_dir, best_acc):
def test(net, test_data, device):
    """
    Performs inference on the validation set

    Parameters
    ----------

    net : nn.Module
        The object representing the network.

    val_loader : torch.utils.data.DataLoader
        The dataloader for the validation set.

    criterion : torch.nn.modules.loss._Loss
        The object representing the loss function.

    batch_size : int

    device : str
        Either 'cpu' or 'cuda', depending on the backend used for computation.

    epoch : int
        The number of the current epoch.

    logger : logging.Logger

    writer : torch.utils.tensorboard.SummaryWriter
        The object used to write Tensorboard events.

    exp_dir : str
        The folder where experiment results (model and log) are saved.

    best_acc : float
        The best value for the accuracy so far obtained with the model.

    Returns
    -------

    float
        The loss computed on the validation set at current epoch.

    float
        The accuracy of the model on the validation set at current epoch.

    float
        The best value for the accuracy so far obtained with the model.

    return test_loss/(batch_idx+1), acc, best_acc
    """

    net.eval()
    # test_loss = 0
    # correct = 0
    # total = 0

    # all_targets = np.array([], dtype=int)
    # all_predicted = np.array([], dtype=int)

    with torch.no_grad():
        # for batch_idx, (inputs, targets) in tqdm(
                # enumerate(val_loader), total=n_batches):

        # TODO move somewhere else
        input_image = Image.open(test_data)

        # inputs, targets = inputs.to(device), targets.to(device)
        inputs = val_transforms(input_image).to(device)
        inputs = inputs.unsqueeze(0)

        outputs = net(inputs)

        # print(nn.Softmax(dim=1)(outputs))

        _, predicted = outputs.max(1)
        # print(predicted.item())

        print(f"Predicted class: {predicted.item()} ({class_map_dict[predicted.item()]})")

        # TODO probably should throw away this stuff here
        # total += targets.size(0)
        # correct += predicted.eq(targets).sum().item()

        # Save all for confusion matrix
        # all_targets = np.hstack(
            # (all_targets, targets.cpu().numpy().astype(int)))

        # all_predicted = np.hstack(
            # (all_predicted, predicted.cpu().numpy().astype(int)))

    # Display confusion matrix
    # cm = confusion_matrix(all_targets, all_predicted)

    # Save confusion matrix
    # np.save(os.path.join(exp_dir, "confusion_matrix_test_latest.npy"), cm)

    # Get detailed stats per class
    # stats_per_class = produce_per_class_stats(
        # all_targets, all_predicted, val_loader.dataset.class_map_dict)

    # cm_pretty = cm2df(cm, val_loader.dataset.class_map_dict)

    # print(cm_pretty)

    # Save checkpoint.
    # acc = 100.*correct/total
    # state = {
        # 'net': net.state_dict(),
        # 'acc': acc,
        # 'epoch': epoch,
    # }

    # Add accuracy on validation set to tb
    # writer.add_scalar("accuracy/val", acc, epoch)

    # Compute balanced accuracy
    # bal_acc = balanced_accuracy_score(all_targets, all_predicted)

    # writer.add_scalar("balanced_accuracy/val", bal_acc, epoch)

    # Display accuracy
    # logger.info("Accuracy on validation set after epoch {}: {}".format(
        # epoch+1, acc))

    return
    # return test_loss/(batch_idx+1), acc, best_acc


def main():  # noqa

    args = parse_args()

    exp_dir = os.path.join('experiments', '{}'.format(args.exp_name))

    os.makedirs(exp_dir, exist_ok=True)

    # Logging
    logger = logging.getLogger(__name__)

    log_file = os.path.join(exp_dir, 'log.log')

    setup_logging(log_path=log_file, log_level=args.log_level, logger=logger)

    logger.info("Experiment started at {}".format(
        datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S")))

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
        logger.info('==> CUDA available, Running on GPU :)')
    else:
        device = 'cpu'
        logger.info('==> CUDA unavailable, Running on CPU :(')

    # Initialize datasets and loaders.
    logger.info('==> Preparing data..')

    # train_ids, val_ids = create_train_val_split(args.data_dir,
                                                # args.train_fraction,
                                                # args.val_fraction)

    # Model.
    logger.info('==> Building model..')

    # Set manually the shape of the last layer so that the same line works
    # for pretrained networks as well.

    # num_classes = train_set.get_num_classes()
    # num_classes = train_loader.dataset.get_num_classes()

    if args.network == 'SimpleCNN':
        net = SimpleCNN(num_classes=num_classes)
        # net = MyCNN(num_classes=num_classes)
    elif args.network == 'Alexnet':
        net = alexnet(pretrained=args.use_pretrained)
        net.classifier[6] = nn.Linear(4096, num_classes)

    elif args.network == 'resnet34':
        # net = resnet34(num_classes=num_classes)

        net = resnet34(pretrained=args.use_pretrained)
        net.fc = nn.Linear(512, num_classes)

    elif args.network == 'resnet50':
        net = resnet50(pretrained=args.use_pretrained)
        net.fc = nn.Linear(2048, num_classes)
        # TODO network specific, move somewhere else
        net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                              bias=False)

    elif args.network == 'resnet101':
        net = resnet101(pretrained=args.use_pretrained)
        net.fc = nn.Linear(2048, num_classes)

    elif args.network == 'resnet152':
        net = resnet152(pretrained=args.use_pretrained)
        net.fc = nn.Linear(2048, num_classes)
    elif args.network == 'VGG16':
        # net = vgg16(num_classes=num_classes)

        net = vgg16(pretrained=args.use_pretrained)
        # TODO check this
        net.classifier[6] = nn.Linear(4096, num_classes)

    # TODO this is required!
    logger.info('==> Loading checpoint for best model..')

    checkpoint = torch.load(
        os.path.join(exp_dir, "model_latest.pth.tar"))
    net.load_state_dict(checkpoint['net'])

    net = net.to(device)

    #################
    ### INFERENCE ###  # noqa
    #################

    tic = time.time()

    # Only this?
    test(net, args.test_data, device)

    toc = time.time()

    dt_val = toc - tic

    logger.info("Inference took {} s.".format(dt_val))


if __name__ == '__main__':

    main()
