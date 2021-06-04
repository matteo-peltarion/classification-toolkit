# import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# def conv2D_output_size(img_size, padding, kernel_size, stride):
    # # compute output shape of conv2D
    # outshape = (
        # np.floor((img_size[0] + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0] + 1).astype(int),  # noqa
        # np.floor((img_size[1] + 2 * padding[1] - (kernel_size[1] - 1) - 1) / stride[1] + 1).astype(int))  # noqa

    # return outshape  # noqa


# def convtrans2D_output_size(img_size, padding, kernel_size, stride):
    # # compute output shape of conv2D
    # outshape = ((img_size[0] - 1) * stride[0] - 2 * padding[0] + kernel_size[0],
                # (img_size[1] - 1) * stride[1] - 2 * padding[1] + kernel_size[1])
    # return outshape


class ResNet_AE(nn.Module):
    """Source: https://github.com/hsinyilin19/ResNetVAE."""
    def __init__(self, fc_hidden1=1024, fc_hidden2=768, drop_p=0.3, CNN_embed_dim=256):
        super(ResNet_AE, self).__init__()

        self.fc_hidden1, self.fc_hidden2, self.CNN_embed_dim = (
            fc_hidden1, fc_hidden2, CNN_embed_dim)

        # CNN architechtures
        self.ch1, self.ch2, self.ch3, self.ch4 = 16, 32, 64, 128

        self.k1, self.k2, self.k3, self.k4 = (
            (5, 5), (3, 3), (3, 3), (3, 3))  # 2d kernal size

        self.s1, self.s2, self.s3, self.s4 = (
            (2, 2), (2, 2), (2, 2), (2, 2))  # 2d strides

        self.pd1, self.pd2, self.pd3, self.pd4 = (
            (0, 0), (0, 0), (0, 0), (0, 0))  # 2d padding

        # encoding components
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.fc1 = nn.Linear(resnet.fc.in_features, self.fc_hidden1)
        self.bn1 = nn.BatchNorm1d(self.fc_hidden1, momentum=0.01)
        self.fc2 = nn.Linear(self.fc_hidden1, self.fc_hidden2)
        self.bn2 = nn.BatchNorm1d(self.fc_hidden2, momentum=0.01)

        # Decoder
        self.convTrans6 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=64, out_channels=32, kernel_size=self.k4,
                stride=self.s4, padding=self.pd4),
            nn.BatchNorm2d(32, momentum=0.01),
            nn.ReLU(inplace=True),
        )

        self.convTrans7 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=32, out_channels=8, kernel_size=self.k3,
                stride=self.s3, padding=self.pd3),
            nn.BatchNorm2d(8, momentum=0.01),
            nn.ReLU(inplace=True),
        )

        self.convTrans8 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=8, out_channels=3, kernel_size=self.k2,
                stride=self.s2, padding=self.pd2),
            nn.BatchNorm2d(3, momentum=0.01),
            nn.Sigmoid()    # y = (y1, y2, y3) \in [0 ,1]^3
        )

    def encode(self, x):
        x = self.resnet(x)  # ResNet
        x = x.view(x.size(0), -1)  # flatten output of conv

        # FC layers
        x = self.bn1(self.fc1(x))
        x = self.relu(x)
        x = self.bn2(self.fc2(x))
        x = self.relu(x)

        return x

    def decode(self, z):

        x = self.convTrans6(z.view(-1, 64, 4, 4))
        x = self.convTrans7(x)
        x = self.convTrans8(x)
        x = F.interpolate(x, size=(224, 224), mode='bilinear')
        return x

    def forward(self, x):
        z = self.encode(x)
        x_reconst = self.decode(z)

        return x_reconst, z


### 👾👾👾 FROM NOW ON CHECK!!!  # noqa

# def llll():
    # # EncoderCNN architecture
    # CNN_fc_hidden1, CNN_fc_hidden2 = 1024, 1024
    # CNN_embed_dim = 256     # latent dim extracted by 2D CNN
    # res_size = 224        # ResNet image size
    # dropout_p = 0.2       # dropout probability
    # # training parameters
    # epochs = 20        # training epochs
    # batch_size = 50
    # learning_rate = 1e-3
    # log_interval = 10   # interval for displaying training info

    # # save model
    # run = 0
    # save_model_path = '../models/results_resnetAE/'
