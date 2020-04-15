# Network options
from torchvision.models.vgg import vgg16  # noqa
from torchvision.models.alexnet import alexnet  # noqa
from torchvision.models.resnet import (  # noqa
    resnet50, resnet34, resnet101, resnet152)

# from palladio.networks.SimpleCNN import SimpleCNN

import torch.nn as nn


def get_network(args, num_classes):

    if args.network == 'resnet50':
        net = resnet50(pretrained=args.use_pretrained)
        net.fc = nn.Linear(2048, num_classes)

        # TODO network specific, move somewhere else
        net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                              bias=False)

    # elif args.network == 'Alexnet':
        # net = alexnet(pretrained=args.use_pretrained)
        # net.classifier[6] = nn.Linear(4096, num_classes)

    # elif args.network == 'resnet34':
        # # net = resnet34(num_classes=num_classes)

        # net = resnet34(pretrained=args.use_pretrained)
        # net.fc = nn.Linear(512, num_classes)

    # elif args.network == 'resnet101':
        # net = resnet101(pretrained=args.use_pretrained)
        # net.fc = nn.Linear(2048, num_classes)

    # elif args.network == 'resnet152':
        # net = resnet152(pretrained=args.use_pretrained)
        # net.fc = nn.Linear(2048, num_classes)
    # elif args.network == 'VGG16':
        # # net = vgg16(num_classes=num_classes)

        # net = vgg16(pretrained=args.use_pretrained)
        # # TODO check this
        # net.classifier[6] = nn.Linear(4096, num_classes)

    # elif args.network == 'SimpleCNN':
        # net = SimpleCNN(num_classes=num_classes)
        # # net = MyCNN(num_classes=num_classes)

    return net
