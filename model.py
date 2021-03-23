import torch.nn as nn
import torch.nn.functional as F


# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        # input size after conv1 is: 28x28x6
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # input size after pool is: 14x14x6
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        # input size after conv2 is: 10x10x16
        # input size after conv2 and pool is: 5x5x16
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
