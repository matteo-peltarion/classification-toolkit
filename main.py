"""Train classifier on HAM10000 dataset.
"""

import os

import time

import datetime

import logging

import argparse

from tqdm import tqdm

import matplotlib

import PIL

matplotlib.use('agg')

from sklearn.metrics import confusion_matrix

import numpy as np

# Torch stuff
import torch.optim as optim
import torch.nn as nn

# from torch.utils.data.sampler import RandomSampler
from torch.utils.data.dataloader import DataLoader

from torch.utils.data import WeightedRandomSampler

from torch.utils.tensorboard import SummaryWriter

import torch
# import torch.backends.cudnn as cudnn

# Network options
from networks.SimpleCNN import SimpleCNN

from torchvision.models.vgg import vgg16
from torchvision.models.alexnet import alexnet
from torchvision.models.resnet import resnet50, resnet34

import torchvision.transforms as transforms

# Experiment specific imports
from lib.dataset import create_train_val_split, HAM10000

from lib.utils import (
    setup_logging, save_checkpoint, create_loss_plot, cm2df,
    produce_per_class_stats)

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

    parser.add_argument('--data-augmentation-level', default=0, type=int,
                        help='Sets different options for DA.')

    parser.add_argument('--batch-size', default=4, type=int,
                        help='batch-size to use')

    parser.add_argument('--lr', default=1e-3, type=float,
                        help='Learning rate')

    parser.add_argument('--milestones', nargs='+', type=int,
                        help='Milestones for lr scheduler')

    parser.add_argument('--network', default='SimpleCNN',
                        choices=[
                            'SimpleCNN', 'VGG16', 'Alexnet', 'resnet34',
                            'resnet50'],
                        help='network architecture')

    parser.add_argument('--num-epochs', default=10, type=int,
                        help='Number of training epochs')

    parser.add_argument('--normalize-input',
                        action='store_true',
                        help='Normalize input using mean and variance computed on training set')  # noqa

    parser.add_argument('--weighted-loss',
                        action='store_true',
                        help='Use a weighted version of the loss.')

    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='Optimizer.')

    parser.add_argument('--resume', '-r',
                        action='store_true', help='resume from checkpoint')

    args = parser.parse_args()

    return args


# Training.
def train(net, train_loader, criterion, optimizer,
          batch_size, device, epoch, logger, writer):
    """Performs training for one epoch
    """

    # TODO move in arguments
    # print message only every N batches
    print_every = 50

    logger.info('Epoch: {}'.format(epoch + 1))

    net.train()

    train_loss = 0
    correct = 0
    total = 0

    # Needs to be changed because of the DL sampler
    # n_batches = len(train_loader.dataset) // batch_size
    n_batches = len(train_loader)

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

        if (batch_idx + 1) % print_every == 0:

            # print("outputs")
            # print(outputs)
            # print("targets")
            # print(targets)
            # for p,n in zip(net.parameters(),net._all_weights[0]):
                # if n[:6] == 'weight':
                    # print('===========\ngradient:{}\n----------\n{}'.format(n,p.grad))

            # Compute accuracy
            acc = 100.*correct/total

            logger.info(constants.STATUS_MSG.format(
                batch_idx+1,
                n_batches,
                train_loss/(batch_idx+1),
                acc))

    # Print gradients
    # for p in net.parameters():
        # print(p)
        # print(p.size())
        # print(p.grad)

    # Add accuracy on validation set to tb
    writer.add_scalar("accuracy/train", acc, epoch)

    return train_loss/(batch_idx+1), acc


def test(net, val_loader, criterion,
         batch_size, device, epoch, logger, writer, exp_dir, best_acc):

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

    # Display confusion matrix
    cm = confusion_matrix(all_targets, all_predicted)

    # Get detailed stats per class
    stats_per_class = produce_per_class_stats(
        all_targets, all_predicted, val_loader.dataset.class_map_dict)

    # for k in val_loader.dataset.class_map_dict:
        # print("Class {}".format(k))
        # print("Precision: {}".format(stats_per_class[k]['precision_score']))
        # print("Recall: {}".format(stats_per_class[k]['recall_score']))
        # print("Roc AUC: {}".format(stats_per_class[k]['roc_auc_score']))

    # Add scalars corresponding to these metrics to tensorboard
    for score in ['precision_score', 'recall_score', 'roc_auc_score']:
        for k in val_loader.dataset.class_map_dict:
            # Add scalars to tb
            writer.add_scalar(
                "{}_{}".format(k, score),
                stats_per_class[k][score],
                epoch)

    cm_pretty = cm2df(cm, val_loader.dataset.class_map_dict)

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


def get_data_augmentation_transforms(level, normalize_input=False):
    """Returns the list of transforms to be applied to the training dataset
       only, for data augmentation.
    """

    # Keep this as an example
    # transform = transforms.Compose([
        # transforms.Resize(256),
        # transforms.RandomCrop(224),
        # transforms.RandomHorizontalFlip(),
        # transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # First create the list of transforms, add each one individually (for
    # better code readability), then return a composition of all the
    # transforms.

    transforms_list = list()

    # Slightly change colors
    if level >= 3:
        colorjitter_transform = transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2)
        transforms_list.append(transforms.RandomApply(
            [colorjitter_transform], 0.5))

    # Horizontal/vertical flip with 0.5 chance
    if level >= 4:
        rotation_transform = transforms.RandomRotation(
            20, resample=PIL.Image.BILINEAR)
        transforms_list.append(
            transforms.RandomApply([rotation_transform], p=0.5))

    # A random resized crop
    if level >= 2:
        crop_transform = transforms.RandomResizedCrop(
            (450, 600), scale=(0.8, 1.0), ratio=(1, 1))
        transforms_list.append(
            transforms.RandomApply([crop_transform], p=0.5))

    # Horizontal/vertical flip with 0.5 chance
    if level >= 1:
        transforms_list.append(transforms.RandomHorizontalFlip(0.5))
        transforms_list.append(transforms.RandomVerticalFlip(0.5))

    transforms_list.append(transforms.ToTensor())

    # Add normalization?
    if normalize_input:
        transforms_list.append(transforms.Normalize(
            constants.NORMALIZATION_MEAN,
            constants.NORMALIZATION_STD))

    return transforms.Compose(transforms_list)


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
    train_ids, val_ids = create_train_val_split(args.data_dir,
                                                args.train_fraction,
                                                args.val_fraction)

    # Load datasets
    train_transforms = get_data_augmentation_transforms(
        args.data_augmentation_level, args.normalize_input)
    train_set = HAM10000(
        args.data_dir, train_ids, transforms=train_transforms)

    # For validation have data augmentation level set to 0 (NO DA)
    val_transforms = get_data_augmentation_transforms(
        0, args.normalize_input)
    val_set = HAM10000(
        args.data_dir, val_ids, transforms=val_transforms)

    weights = train_set.make_weights_for_balanced_classes()

    # train_sampler = RandomSampler(train_set)

    # N_samples_per_epoch = len(weights)
    # This one is sort of eyeballed based on a few assumptions
    # TODO explain
    N_samples_per_epoch = 50 * 7 * args.batch_size

    # For finding appropriate lr
    # N_samples_per_epoch = 50 * 7
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
        # net = MyCNN(num_classes=num_classes)
    elif args.network == 'Alexnet':
        net = alexnet(pretrained=False, num_classes=num_classes)
    elif args.network == 'resnet34':
        net = resnet34(num_classes=num_classes)
    elif args.network == 'resnet50':
        net = resnet50(num_classes=num_classes)
    elif args.network == 'VGG16':
        net = vgg16(num_classes=num_classes)

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
        start_epoch = checkpoint['epoch']
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
    if args.weighted_loss:
        loss_weights = torch.Tensor(list(constants.CLASSES_WEIGHTS.values()))
        loss_weights = loss_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=loss_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = None
    if args.optimizer == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=args.lr)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)

    # Check
    assert(optimizer is not None)

    if args.milestones is None:
        milestones = []
    else:
        milestones = args.milestones

    # Set lr scheduler
    lr_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=0.3162)  # sqrt(0.1)

    epochs, train_losses, test_losses = [], [], []

    # Keep an eye on how long it takes to train for one epoch
    times_train = list()
    times_val = list()

    # Training loop
    for epoch in range(start_epoch, args.num_epochs):

        tic = time.time()

        # Print learning rate
        for param_group in optimizer.param_groups:
            logger.info('Learning rate: {}'.format(param_group['lr']))

        # Train for one epoch
        train_loss, train_acc = train(
            net, train_loader, criterion, optimizer,
            args.batch_size, device, epoch, logger, writer)

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

        logger.info("ETA: {}".format(eta.strftime("%d/%m/%Y, %H:%M:%S")))

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
