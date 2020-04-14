#!/usr/bin/env python
"""Train classifier on Husqvarna dataset.
"""

import os

import time

import datetime

import logging

import argparse

from tqdm import tqdm

import matplotlib

matplotlib.use('agg')

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, balanced_accuracy_score

import numpy as np

# Torch stuff
import torch.optim as optim
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter

import torch
# import torch.backends.cudnn as cudnn

# Network options
from palladio.networks.SimpleCNN import SimpleCNN

from torchvision.models.vgg import vgg16
from torchvision.models.alexnet import alexnet
from torchvision.models.resnet import (
    resnet50, resnet34, resnet101, resnet152)

from lib.utils import (
    setup_logging, save_checkpoint, create_loss_plot, cm2df,
    produce_per_class_stats)

# Collect constants in separate file, C headers style.
import constants

# Required for loading configuration dynamically
import importlib.util

# global
class_map_dict = None


def parse_args():

    # Parse args.
    parser = argparse.ArgumentParser(
        description='PyTorch classifier on custom dataset')

    # TODO move to konfiguration probably
    # parser.add_argument('--data-dir', help='Path to data', required=True)

    parser.add_argument('--train_fraction', default=0.8, type=float,
                        help='fraction of dataset to use for training')

    parser.add_argument('--val_fraction', default=0.2, type=float,
                        help='fraction of dataset to use for validation')

    # parser.add_argument('--input-features', nargs='+', type=str,
                        # help='List of columns where inpute features are',
                        # metavar='FEATURE_NAME',
                        # required=True)

    parser.add_argument('--exp-name', default='baseline', type=str,
                        help='name of experiment')

    parser.add_argument('--log-level', default='INFO',
                        choices=['DEBUG', 'INFO'], help='log-level to use')

    parser.add_argument('--data-augmentation-level', default=0, type=int,
                        help='Sets different options for DA.')

    parser.add_argument('--batch-size', default=4, type=int,
                        help='batch-size to use')

    parser.add_argument('--lr', default=1e-3, type=float,
                        help='Learning rate')

    parser.add_argument('--weight-decay', default=0.0, type=float,
                        help='Weight decay parameter for optimizer.')

    parser.add_argument('--scheduler-gamma', default=0.1, type=float,
                        help='Gamma parameter for learning rate scheduler.')

    parser.add_argument('--class-weights', nargs='+', type=float,
                        help='Weights for class (used for loss)')

    parser.add_argument('--milestones', nargs='+', type=int,
                        help='Milestones for lr scheduler')

    parser.add_argument('--network', default='SimpleCNN',
                        choices=[
                            'SimpleCNN', 'VGG16', 'Alexnet',
                            'resnet34', 'resnet50', 'resnet101', 'resnet152'],
                        help='network architecture')

    parser.add_argument('--use-pretrained',
                        action='store_true',
                        help='Use a pretrained model.')  # noqa

    parser.add_argument('--num-epochs', default=10, type=int,
                        help='Number of training epochs')

    parser.add_argument('--lr-finder',
                        action='store_true',
                        help='Exploratory run for finding best lr finder.')  # noqa

    parser.add_argument('--normalize-input',
                        action='store_true',
                        help='Normalize input using mean and variance computed on training set')  # noqa

    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='Optimizer.')

    parser.add_argument('--konfiguration', '-K', type=str,
                        default='konfiguration.py',
                        help='Path to file containing configuration.')

    parser.add_argument('--resume', '-r',
                        action='store_true', help='resume from checkpoint')

    args = parser.parse_args()

    return args


# Training.
def train(net, train_loader, criterion, optimizer,
          batch_size, device, epoch, logger, writer, exp_dir,
          print_every=50):
    """Performs training for one epoch

    print_every : int
        Print message only every N batches. Default 50.
    """

    logger.info('Epoch: {}'.format(epoch + 1))

    net.train()

    train_loss = 0
    correct = 0
    total = 0

    # Needs to be changed because of the DL sampler
    # n_batches = len(train_loader.dataset) // batch_size
    n_batches = len(train_loader)

    all_targets = np.array([], dtype=int)
    all_predicted = np.array([], dtype=int)

    for batch_idx, (inputs, targets) in enumerate(train_loader):

        # Send tensors to the appropriate device
        inputs, targets = inputs.to(device), targets.to(device)

        # Set gradients to 0
        optimizer.zero_grad()

        # Forward pass
        outputs = net(inputs)

        # Compute loss
        loss = criterion(outputs, targets)

        # Backprop
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        _, predicted = outputs.max(1)

        total += targets.size(0)

        correct += predicted.eq(targets).sum().item()

        # Compute accuracy
        acc = 100.*correct/total

        if (batch_idx + 1) % print_every == 0:
            logger.info(constants.STATUS_MSG.format(
                batch_idx+1,
                n_batches,
                train_loss/(batch_idx+1),
                acc))

        # TODO add this to parameter
        if (epoch + 1) % 10 == 0:
            # Save all for confusion matrix
            all_targets = np.hstack(
                (all_targets, targets.cpu().numpy().astype(int)))

            all_predicted = np.hstack(
                (all_predicted, predicted.cpu().numpy().astype(int)))

    # TODO add this to parameter
    if (epoch + 1) % 10 == 0:
        # Display confusion matrix
        cm = confusion_matrix(all_targets, all_predicted)

        # Save confusion matrix
        np.save(os.path.join(exp_dir, "confusion_matrix_train_latest.npy"), cm)

        # TODO check what happens if class_map_dict is not set (also for
        # validation)

        # Get detailed stats per class
        stats_per_class = produce_per_class_stats(
            all_targets, all_predicted, class_map_dict)

        # Add scalars corresponding to these metrics to tensorboard
        for score in ['precision_score', 'recall_score',
                      'roc_auc_score']:
            for k in class_map_dict:
                # Add scalars to tb
                writer.add_scalar(
                    "{}_{}_train".format(k, score),
                    stats_per_class[k][score],
                    epoch)

        cm_pretty = cm2df(cm, class_map_dict)

        print(cm_pretty)

        # Compute balanced accuracy
        bal_acc = balanced_accuracy_score(all_targets, all_predicted)

        writer.add_scalar("balanced_accuracy/train", bal_acc, epoch)

    # Add accuracy on validation set to tb
    writer.add_scalar("accuracy/train", acc, epoch)

    return train_loss/(batch_idx+1), acc


def test(net, val_loader, criterion,
         batch_size, device, epoch, logger, writer, exp_dir, best_acc):
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
    test_loss = 0
    correct = 0
    total = 0
    # n_batches = len(val_loader.dataset) // batch_size
    n_batches = len(val_loader)

    all_targets = np.array([], dtype=int)
    all_predicted = np.array([], dtype=int)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in tqdm(
                enumerate(val_loader), total=n_batches):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)

            # TODO probably should throw away this stuff here
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Save all for confusion matrix
            all_targets = np.hstack(
                (all_targets, targets.cpu().numpy().astype(int)))

            all_predicted = np.hstack(
                (all_predicted, predicted.cpu().numpy().astype(int)))

    # Display confusion matrix
    cm = confusion_matrix(all_targets, all_predicted)

    # Save confusion matrix
    np.save(os.path.join(exp_dir, "confusion_matrix_test_latest.npy"), cm)

    # Get detailed stats per class
    stats_per_class = produce_per_class_stats(
        all_targets, all_predicted, class_map_dict)

    # Add scalars corresponding to these metrics to tensorboard
    for score in ['precision_score', 'recall_score', 'roc_auc_score']:
        for k in class_map_dict:
            # Add scalars to tb
            writer.add_scalar(
                "{}_{}_val".format(k, score),
                stats_per_class[k][score],
                epoch)

    cm_pretty = cm2df(cm, class_map_dict)

    print(cm_pretty)

    # Save checkpoint.
    acc = 100.*correct/total
    state = {
        'net': net.state_dict(),
        'acc': acc,
        'epoch': epoch,
    }

    # Add accuracy on validation set to tb
    writer.add_scalar("accuracy/val", acc, epoch)

    # Compute balanced accuracy
    bal_acc = balanced_accuracy_score(all_targets, all_predicted)

    writer.add_scalar("balanced_accuracy/val", bal_acc, epoch)

    # Display accuracy
    logger.info("Accuracy on validation set after epoch {}: {}".format(
        epoch+1, acc))

    if acc > best_acc:
        logger.info('Saving..')
        save_checkpoint(state, exp_dir, backup_as_best=True)
        best_acc = acc
    else:
        save_checkpoint(state, exp_dir, backup_as_best=False)

    return test_loss/(batch_idx+1), acc, best_acc


def main():  # noqa

    args = parse_args()

    exp_dir = os.path.join('experiments', '{}'.format(args.exp_name))

    os.makedirs(exp_dir, exist_ok=True)

    # Writer will output to ./runs/ directory by default
    # writer = SummaryWriter(os.path.join(exp_dir, "tb_log"))
    writer = SummaryWriter(comment=args.exp_name)

    # Logging
    logger = logging.getLogger(__name__)

    log_file = os.path.join(exp_dir, 'log.log')

    setup_logging(log_path=log_file, log_level=args.log_level, logger=logger)

    # Keep track of how much the whole experiment lasts
    experiment_start = time.time()

    logger.info("Experiment started at {}".format(
        datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S")))

    # Globals. NOPE

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
        logger.info('==> CUDA available, Running on GPU :)')
    else:
        device = 'cpu'
        logger.info('==> CUDA unavailable, Running on CPU :(')

    npy_file = os.path.join(exp_dir, 'final_results.npy')

    # Dump arguments in text file inside the experiment folder
    args_file = os.path.join(exp_dir, 'args.log')

    with open(args_file, 'w') as the_file:
        the_file.write(str(args))

    # Initialize datasets and loaders.
    logger.info('==> Preparing data..')

    spec = importlib.util.spec_from_file_location(
        "konfiguration",
        "konfiguration.py")
    konfiguration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(konfiguration)

    # Import experiment specific stuff
    # from konfiguration import (
        # train_loader, val_loader, num_classes, _class_map_dict)

    # Manually load stuf from konfiguration
    train_loader = konfiguration.train_loader
    val_loader = konfiguration.val_loader
    num_classes = konfiguration.num_classes
    _class_map_dict = konfiguration.class_map_dict

    global class_map_dict
    class_map_dict = _class_map_dict

    # Model.
    logger.info('==> Building model..')

    # Set manually the shape of the last layer so that the same line works
    # for pretrained networks as well.

    if args.use_pretrained:
        # Load checkpoint.
        logger.info('==> Using pretrained model..')

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

    if args.resume:
        # Load checkpoint.
        logger.info('==> Resuming from checkpoint..')
        # assert (
            # os.path.isdir('checkpoint'),
            # 'Error: no checkpoint directory found!')
        checkpoint = torch.load(
            os.path.join(exp_dir, "model_latest.pth.tar"))
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch'] + 1
    else:
        best_acc = 0
        start_epoch = 0

    net = net.to(device)

    # You seem weird, let's shush you
    # if device == 'cuda':
        # #
        # net = torch.nn.DataParallel(net)
        # cudnn.benchmark = True

    # Define the loss
    if args.class_weights is not None:
        assert (
            len(args.class_weights) == train_loader.dataset.get_num_classes())
        logger.info('==> Using class weights for loss:')
        logger.info("==> {}".format(args.class_weights))
        loss_weights = torch.Tensor(args.class_weights)
        loss_weights = loss_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=loss_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = None
    if args.optimizer == 'Adam':
        optimizer = optim.Adam(
            net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'SGD':
        # TODO momentum might become a parameter?
        optimizer = optim.SGD(
            net.parameters(), lr=args.lr, momentum=0.9,
            weight_decay=args.weight_decay)

    if args.milestones is None:
        milestones = []
    else:
        milestones = args.milestones

    # Learning rate scheduler must not be set when running the lr finder
    if not args.lr_finder:

        # Set lr scheduler
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones,
            last_epoch=start_epoch-1,
            gamma=args.scheduler_gamma)  # default: 0.1
            # gamma=0.3162)  # sqrt(0.1)

    epochs, train_losses, test_losses = [], [], []

    # Keep an eye on how long it takes to train for one epoch
    times_train = list()
    times_val = list()

    #################
    ### LR FINDER ###  # noqa
    #################

    if args.lr_finder:

        # Import is here so that it doesn't mess with logging
        from torch_lr_finder import LRFinder

        # Try up to two orders of magnitude greater
        end_lr = 100*args.lr

        print(f"Trying 100 values between {args.lr} and {end_lr}")

        # TODO Move this stuff in separate function?
        lr_finder = LRFinder(net, optimizer, criterion, device="cuda")
        # lr_finder.range_test(train_loader, end_lr=1e-2, num_iter=2)
        lr_finder.range_test(train_loader, end_lr=end_lr, num_iter=100)

        # print(lr_finder.history)

        # Plot loss against lr
        fig, ax = plt.subplots()

        ax.plot(lr_finder.history["lr"], lr_finder.history["loss"])
        ax.set_xscale('log')

        # Plot title and axis labels
        ax.set_title("LR finder results")
        ax.set_xlabel("lr")
        ax.set_ylabel("Loss")

        plt.savefig("lr_finder_output.png")
        return

    ################
    ### TRAINING ###  # noqa
    ################

    # Training loop
    for epoch in range(start_epoch, args.num_epochs):

        tic = time.time()

        # Print the experiment name at the beginning of every loop
        logger.info("Experiment name: {}".format(args.exp_name))

        # Print learning rate
        for param_group in optimizer.param_groups:
            logger.info('Learning rate: {}'.format(param_group['lr']))

        # Train for one epoch
        train_loss, train_acc = train(
            net, train_loader, criterion, optimizer,
            args.batch_size, device, epoch, logger, writer, exp_dir)

        toc = time.time()

        dt_train = toc - tic
        times_train.append(dt_train)
        train_losses.append(train_loss)

        logger.info("Training for epoch {} took {} s.".format(
            epoch+1, dt_train))

        # Reset tic for test
        tic = toc

        # Test results
        test_loss, test_acc, best_acc = test(
            net, val_loader, criterion,
            args.batch_size, device, epoch,
            logger, writer, exp_dir, best_acc)

        toc = time.time()

        dt_val = toc - tic
        test_losses.append(test_loss)
        times_val.append(dt_val)

        logger.info("Validation for epoch {} took {} s.".format(
            epoch+1, dt_val))

        epochs.append(epoch)

        # Estimate how long each epoch takes
        estimated_seconds_per_epoch = (toc - experiment_start)/(epoch+1)

        logger.info("Elapsed time after epoch {}: {} ({} per epoch)".format(
            epoch+1,
            datetime.timedelta(
                seconds=int(toc - experiment_start)),
            datetime.timedelta(
                seconds=int(estimated_seconds_per_epoch)),
        ))

        # Estimate ETA
        eta = (datetime.datetime.now() +
               datetime.timedelta(
                   seconds=(args.num_epochs - epoch - 1) *
                   estimated_seconds_per_epoch))

        logger.info("ETA: {}".format(eta.strftime("%d/%m/%Y %H:%M:%S")))

        # Add scalars to tb
        # Loss
        writer.add_scalars(
            'Loss', {
                'train': train_loss,
                'val': test_loss},
            epoch)

        # Accuracy
        writer.add_scalars(
            'Accuracy', {
                'train': train_acc,
                'val': test_acc},
            epoch)

        create_loss_plot(exp_dir, epochs, train_losses, test_losses)

        np.save(npy_file, [train_losses, test_losses])

        lr_scheduler.step()

    # Keep track of how much the whole experiment lasts
    experiment_end = time.time()

    logger.info("Experiment ended at {}".format(
        datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S")))

    logger.info("Elapsed time: {}".format(
        datetime.timedelta(seconds=int(experiment_end - experiment_start))))

    writer.close()


if __name__ == '__main__':

    main()
