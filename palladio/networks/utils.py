# Network options
from torchvision.models.vgg import vgg16  # noqa
from torchvision.models.alexnet import alexnet  # noqa
from torchvision.models.resnet import (  # noqa
    resnet18, resnet34, resnet50, resnet101, resnet152)

from torchvision.models.densenet import (  # noqa
    densenet121)

# from efficientnet_pytorch import EfficientNet

import torch.nn as nn


def get_network(network_name, num_classes, use_pretrained, n_input_channels=3):

    # if network_name == 'efficientnet-b0':
        # if use_pretrained:
            # net = EfficientNet.from_pretrained(
                # 'efficientnet-b0', in_channels=n_input_channels,
                # num_classes=num_classes)
        # else:
            # net = EfficientNet.from_name(
                # 'efficientnet-b0', in_channels=n_input_channels,
                # num_classes=num_classes)

    # if network_name == 'efficientnet-b4':
        # if use_pretrained:
            # net = EfficientNet.from_pretrained(
                # 'efficientnet-b4', in_channels=n_input_channels,
                # num_classes=num_classes)
        # else:
            # net = EfficientNet.from_name(
                # 'efficientnet-b4', in_channels=n_input_channels,
                # num_classes=num_classes)

    # if network_name == 'efficientnet-b7':
        # if use_pretrained:
            # net = EfficientNet.from_pretrained(
                # 'efficientnet-b7', in_channels=n_input_channels,
                # num_classes=num_classes)
        # else:
            # net = EfficientNet.from_name(
                # 'efficientnet-b7', in_channels=n_input_channels,
                # num_classes=num_classes)

    if network_name.startswith("densenet"):
        name_class_map = {
            'densenet121': densenet121,
        }

        torchvision_class = name_class_map[network_name]

        net = torchvision_class(pretrained=use_pretrained)
        net.classifier = nn.Linear(net.classifier.in_features, num_classes)

        # If the number of input channels is != 3, adapt network
        if n_input_channels != 3:
            net.features.conv0 = nn.Conv2d(
                n_input_channels, 64, kernel_size=7, stride=2, padding=3,
                bias=False)

    if network_name.startswith("resnet"):
        name_class_map = {
            'resnet18': resnet18,
            'resnet34': resnet34,
            'resnet50': resnet50,
            'resnet101': resnet101,
        }

        torchvision_class = name_class_map[network_name]

        net = torchvision_class(pretrained=use_pretrained)
        net.fc = nn.Linear(net.fc.in_features, num_classes)

        # If the number of input channels is != 3, adapt network
        if n_input_channels != 3:
            net.conv1 = nn.Conv2d(
                n_input_channels, 64, kernel_size=7, stride=2, padding=3,
                bias=False)

    # Distributed, when required
    # net = nn.DataParallel(net)

    return net
