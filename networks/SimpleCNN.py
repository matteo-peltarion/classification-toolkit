import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):

    # TODO add input shape as parameter and adjust the rest
    def __init__(self, num_classes=1000):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.conv1.weight)

        self.pool = nn.MaxPool2d(kernel_size=2, padding=0)

        # TODO architecture should adjust to input
        # self.fc1 = nn.Linear(32 * 56 * 75, 64)
        self.fc1 = nn.Linear(24 * 24 * 32, 64)
        nn.init.xavier_uniform_(self.fc1.weight)

        self.fc2 = nn.Linear(64, num_classes)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        # x = x.view(-1, 32 * 56 * 75)
        x = x.view(-1, 24 * 24 * 32)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return(x)
