import torch.nn as nn
import torch.nn.functional as F


class MyCNN(nn.Module):

    def __init__(self, num_classes=1000):
        super(MyCNN, self).__init__()

        # input = 600 * 450

        # 32
        self.conv11 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        nn.init.xavier_uniform(self.conv11.weight)

        self.conv12 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        nn.init.xavier_uniform(self.conv12.weight)

        self.pool1 = nn.MaxPool2d(kernel_size=2, padding=0)

        # 64
        self.conv21 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        nn.init.xavier_uniform(self.conv21.weight)

        self.conv22 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        nn.init.xavier_uniform(self.conv22.weight)

        self.pool2 = nn.MaxPool2d(kernel_size=2, padding=0)

        # 128
        self.conv31 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        nn.init.xavier_uniform(self.conv31.weight)

        self.conv32 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        nn.init.xavier_uniform(self.conv32.weight)

        self.pool3 = nn.MaxPool2d(kernel_size=2, padding=0)

        self.fc1 = nn.Linear(128 * 56 * 75, 512)
        nn.init.xavier_uniform(self.fc1.weight)
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(512, 512)
        nn.init.xavier_uniform(self.fc2.weight)
        self.dropout2 = nn.Dropout(0.5)

        self.logits = nn.Linear(512, num_classes)
        nn.init.xavier_uniform(self.logits.weight)

    def forward(self, x):

        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = self.pool1(x)

        x = F.relu(self.conv21(x))
        x = F.relu(self.conv22(x))
        x = self.pool2(x)

        x = F.relu(self.conv31(x))
        x = F.relu(self.conv32(x))
        x = self.pool3(x)

        # FC layers
        x = x.view(-1, 128 * 56 * 75)

        x = F.relu(self.fc1(x))
        x = self.dropout1(x)

        x = F.relu(self.fc2(x))
        x = self.dropout2(x)

        x = self.logits(x)
        return(x)


class SimpleCNN(nn.Module):

    def __init__(self, num_classes=1000):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        nn.init.xavier_uniform(self.conv1.weight)

        self.pool = nn.MaxPool2d(kernel_size=8, padding=0)

        self.fc1 = nn.Linear(32 * 56 * 75, 64)
        nn.init.xavier_uniform(self.fc1.weight)

        self.fc2 = nn.Linear(64, num_classes)
        nn.init.xavier_uniform(self.fc2.weight)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(-1, 32 * 56 * 75)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return(x)
