import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
from torch.utils.data import SubsetRandomSampler

from model import ConvNet
from pedestrian_dataset import PedestrianDataset

# set CPU or GPU, if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

dataset = PedestrianDataset(csv_file='DaimlerBenchmark/pedestrian_dataset.csv',
                            root_dir='./',
                            transform=transforms.ToTensor())

train_loader, validation_loader, test_loader = dataset.loader(batch_size=64,
                                                              train_split=.8,
                                                              validation_split=.2,
                                                              shuffle_dataset=True,
                                                              random_seed=0)

# Hyper-parameters
num_epochs = 1
learning_rate = 0.001

model = ConvNet().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
loss_path = np.zeros(shape=(num_epochs, total_step))
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss_path[epoch][i] = loss.item()

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 5 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

plt.figure()
plt.plot(loss_path.ravel())
plt.title('Loss')
plt.xlabel("iteration")
plt.show()

# Test the model
# eval mode (batchnorm uses moving mean/var instead of mini-batch mean/var)
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'
          .format(100.0 * correct / total))

print("")
