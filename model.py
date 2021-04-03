import torch.nn as nn
import torch

# Convolutional neural network (two convolutional layers)


class ConvNet0(nn.Module):
    def __init__(self, num_classes=2):
        super(ConvNet0, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(8*8*32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


class ConvNet1(nn.Module):
    def __init__(self, num_classes=2):
        super(ConvNet1, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc1 = nn.Linear(4 * 4 * 64, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        conv0 = self.layer1(x)
        conv1 = self.layer2(conv0)
        conv2 = self.layer3(conv1)
        conv2_flattened = conv2.reshape(conv2.size(0), -1)
        out1 = self.fc1(conv2_flattened)
        out2 = self.fc2(out1)
        return (conv0, conv1, conv2, conv2_flattened, out1, out2)


