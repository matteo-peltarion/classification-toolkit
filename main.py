"""Train classifier on HAM10000 dataset.
"""

import os

import logging

import argparse

import matplotlib

matplotlib.use('agg')

import torch.optim as optim
import torch.nn as nn

import numpy as np

# from torch.utils.data.sampler import RandomSampler
from torch.utils.data.dataloader import DataLoader

from torch.utils.data import WeightedRandomSampler

# import torch.backends.cudnn as cudnn

from networks.SimpleCNN import SimpleCNN

import torch

from lib.dataset import create_train_val_split, HAM10000

from lib.utils import setup_logging, save_checkpoint, create_loss_plot

# Collect constants in separate file, C headers style.
import constants


def parse_args():

    # Parse args.
    parser = argparse.ArgumentParser(
        description='PyTorch classifier on HAM10000 dataset')

    parser.add_argument('--data-dir', default='./data', help='path to data')

    parser.add_argument('--train_fraction', default=0.8, type=float,
                        help='fraction of dataset to use for training')

    parser.add_argument('--val_fraction', default=0.2, type=float,
                        help='fraction of dataset to use for validation')

    parser.add_argument('--exp-name', default='baseline', type=str,
                        help='name of experiment')

    parser.add_argument('--log-level', default='INFO',
                        choices=['DEBUG', 'INFO'], help='log-level to use')

    parser.add_argument('--batch-size', default=4, type=int,
                        help='batch-size to use')

    parser.add_argument('--network', default='SimpleCNN',
                        choices=['SimpleCNN'],
                        help='network architecture')

    parser.add_argument('--num-epochs', default=10, type=int,
                        help='Number of training epochs')

    args = parser.parse_args()

    return args


# Training.
def train(net, train_loader, criterion, optimizer,
          batch_size, device, epoch, logger):
    """Performs training for one epoch
    """

    logger.info('Epoch: %d' % epoch)

    net.train()

    train_loss = 0
    correct = 0
    total = 0

    n_batches = len(train_loader.dataset) // batch_size

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        logger.info(constants.STATUS_MSG.format(
            batch_idx+1,
            n_batches,
            train_loss/(batch_idx+1),
            100.*correct/total))

    return train_loss/(batch_idx+1)


def test(net, val_loader, criterion,
         batch_size, device, epoch, logger, exp_dir, best_acc):

    # TODO nuke this
    # global best_acc

    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    n_batches = len(val_loader.dataset) // batch_size

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            logger.info(constants.STATUS_MSG.format(batch_idx+1,
                        n_batches,
                        test_loss/(batch_idx+1),
                        100.*correct/total))

    # Save checkpoint.
    acc = 100.*correct/total
    state = {
        'net': net.state_dict(),
        'acc': acc,
        'epoch': epoch,
    }

    if acc > best_acc:
        logger.info('Saving..')
        save_checkpoint(state, exp_dir, backup_as_best=True)
        best_acc = acc
    else:
        save_checkpoint(state, exp_dir, backup_as_best=False)

    return test_loss/(batch_idx+1), best_acc


def main():

    args = parse_args()

    exp_dir = os.path.join('experiments', '{}'.format(args.exp_name))

    os.makedirs(exp_dir, exist_ok=True)

    # Logging
    logger = logging.getLogger(__name__)

    log_file = os.path.join(exp_dir, 'log.log')

    setup_logging(log_path=log_file, log_level=args.log_level, logger=logger)

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
    train_ids, val_ids = create_train_val_split(args.data_dir,
                                                args.train_fraction,
                                                args.val_fraction)

    # Load datasets
    # TODO add transforms for data augmentation for train_set HHH
    train_set = HAM10000(args.data_dir, train_ids)
    val_set = HAM10000(args.data_dir, val_ids)

    weights = train_set.make_weights_for_balanced_classes()

    # train_sampler = RandomSampler(train_set)
    train_sampler = WeightedRandomSampler(weights, len(weights))

    num_classes = train_set.get_num_classes()

    # TODO set num_workers from args
    train_loader = DataLoader(train_set,
                              batch_size=args.batch_size,
                              sampler=train_sampler,
                              num_workers=8)

    val_loader = DataLoader(val_set,
                            batch_size=args.batch_size,
                            num_workers=8)

    # Model.
    logger.info('==> Building model..')

    if args.network == 'SimpleCNN':
        net = SimpleCNN(num_classes=num_classes)

    net = net.to(device)

    # You seem weird, let's shush you
    # if device == 'cuda':
        # #
        # net = torch.nn.DataParallel(net)
        # cudnn.benchmark = True

    # Define the loss
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.Adam(net.parameters())

    epochs, train_losses, test_losses = [], [], []

    # Stop here, for debugging
    # return

    # Init best acc
    best_acc = 0  # best test accuracy

    for epoch in range(0, args.num_epochs):

        # Train for one epoch
        # TODO fix this
        # train_loss = train(epoch)

        # train_losses.append(train_loss)

        # Test results
        # test_loss = test(epoch)
        test_loss, best_acc = test(
            net, val_loader, criterion,
            args.batch_size, device, epoch,
            logger, exp_dir, best_acc)

        test_losses.append(test_loss)

        epochs.append(epoch)

        create_loss_plot(exp_dir, epochs, train_losses, test_losses)

        np.save(npy_file, [train_losses, test_losses])


if __name__ == '__main__':

    main()
