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
                            root_dir='./')
                            # transform=transforms.ToTensor())

batch_size = 1
train_split = .8
validation_split = .2
shuffle_dataset = True
random_seed = 0

# Creating data indices for training and validation splits:
dataset_size = len(dataset)     # 49000

train_size = int(np.floor(train_split * dataset_size))
print("Train size + Val size = ", train_size)
test_size = int(dataset_size - train_size)
train_size = int(np.floor(train_size * (1 - validation_split)))
validation_size = int(dataset_size - train_size - test_size)

print("train_size = ", train_size)
print("validation_size = ", validation_size)
print("test_size", test_size)
print("Sum = ", train_size + validation_size + test_size)

assert train_size + validation_size + test_size == dataset_size

# Train size + Val size =  39200
# train_size =  31360
# validation_size =  7840
# test_size 9800

indices = list(range(dataset_size))
if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices, test_indices = indices[:train_size], \
                                           indices[train_size:train_size + validation_size], \
                                           indices[train_size + validation_size:]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)
test_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)
test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

# Hyper-parameters
num_epochs = 2
learning_rate = 0.001

model = ConvNet().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
loss_path = np.zeros(shape=(num_epochs,total_step))
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

        if (i+1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

print("")