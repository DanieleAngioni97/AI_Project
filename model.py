import torch.nn as nn
import torch

# Convolutional neural network (four convolutional layers)


class ConvNet(nn.Module):
    def __init__(self, num_classes=2):
        super(ConvNet, self).__init__()
        # input 128x64x3
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2, stride=2))  # 64x32x16
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2))  # 32x16x32
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2))  # 16x8x64
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2))  # 8*4*x128
        self.fc1 = nn.Linear(8*4*128, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        conv0 = self.layer1(x)
        conv1 = self.layer2(conv0)
        conv2 = self.layer3(conv1)
        conv3 = self.layer4(conv2)
        conv3_flattened = conv3.view(-1, 8*4*128)
        out1 = self.fc1(conv3_flattened)
        out2 = self.fc2(out1)
        return (conv0, conv1, conv2, conv3, out1, out2)


