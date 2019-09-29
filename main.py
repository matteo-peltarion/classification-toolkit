"""Train classifier on HAM10000 dataset.
"""

import os

import time

import logging

import argparse

from tqdm import tqdm

import matplotlib

matplotlib.use('agg')

from sklearn.metrics import confusion_matrix

import numpy as np

# Torch stuff
import torch.optim as optim
import torch.nn as nn

# from torch.utils.data.sampler import RandomSampler
from torch.utils.data.dataloader import DataLoader

from torch.utils.data import WeightedRandomSampler

import torch
# import torch.backends.cudnn as cudnn

from networks.SimpleCNN import SimpleCNN

# Experiment specific imports
from lib.dataset import create_train_val_split, HAM10000

from lib.utils import setup_logging, save_checkpoint, create_loss_plot, cm2df

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

    # TODO move in arguments
    # print message only every N batches
    print_every = 50

    logger.info('Epoch: %d' % epoch)

    net.train()

    train_loss = 0
    correct = 0
    total = 0

    # Needs to be changed because of the DL sampler
    # n_batches = len(train_loader.dataset) // batch_size
    n_batches = len(train_loader)

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

        if (batch_idx + 1) % print_every == 0:
            logger.info(constants.STATUS_MSG.format(
                batch_idx+1,
                n_batches,
                train_loss/(batch_idx+1),
                100.*correct/total))

    return train_loss/(batch_idx+1)


def test(net, val_loader, criterion,
         batch_size, device, epoch, logger, exp_dir, best_acc):

    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    n_batches = len(val_loader.dataset) // batch_size

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

            # if (batch_idx + 1) % print_every == 0:
                # logger.info(constants.STATUS_MSG.format(batch_idx+1,
                            # n_batches,
                            # test_loss/(batch_idx+1),
                            # 100.*correct/total))

    acc2 = 100*(all_predicted == all_targets).sum()/total

    # Display confusion matrix
    cm = confusion_matrix(all_targets, all_predicted)

    cm_pretty = cm2df(cm, val_loader.dataset.class_map_dict)

    print(cm_pretty)

    # Save checkpoint.
    acc = 100.*correct/total
    state = {
        'net': net.state_dict(),
        'acc': acc,
        'epoch': epoch,
    }

    # Display accuracy
    logger.info("Accuracy on validation set after epoch {}: {}".format(
        epoch, acc))
    logger.info("Accuracy on validation set after epoch {}: {}".format(
        epoch, acc2))

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

    # N_samples_per_epoch = len(weights)
    # This one is sort of eyballed based on a few assumptions
    # TODO explain
    N_samples_per_epoch = 280 * 7
    train_sampler = WeightedRandomSampler(weights, N_samples_per_epoch)

    num_classes = train_set.get_num_classes()

    # Use custom sampler for train_loader
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

    # Keep an eye on how long it takes to train for one epoch
    times_train = list()
    times_val = list()

    for epoch in range(0, args.num_epochs):

        tic = time.time()

        # Train for one epoch
        train_loss = train(
            net, train_loader, criterion, optimizer,
            args.batch_size, device, epoch, logger)

        toc = time.time()

        dt_train = toc - tic
        times_train.append(dt_train)
        train_losses.append(train_loss)

        logger.info("Training for epoch {} took {} s.".format(
            epoch, dt_train))

        # Reset tic for test
        tic = toc

        # Test results
        # test_loss = test(epoch)
        test_loss, best_acc = test(
            net, val_loader, criterion,
            args.batch_size, device, epoch,
            logger, exp_dir, best_acc)

        toc = time.time()

        dt_val = toc - tic
        test_losses.append(test_loss)
        times_val.append(dt_val)

        logger.info("Validation for epoch {} took {} s.".format(
            epoch, dt_val))

        epochs.append(epoch)

        create_loss_plot(exp_dir, epochs, train_losses, test_losses)

        np.save(npy_file, [train_losses, test_losses])


if __name__ == '__main__':

    main()
