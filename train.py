import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
from torch.utils.data import SubsetRandomSampler

from model import ConvNet1, ConvNet0
from pedestrian_dataset import PedestrianDataset
import utils


batch_size = 128
validation_split = 0.2

# Hyper-parameters
num_epochs = 10
learning_rate = 0.001

model_name = "modello_ConvNet0"

# set CPU or GPU, if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

dataset = PedestrianDataset(train=True,
                            transform=transforms.ToTensor())


torch.manual_seed(0)

model = ConvNet0().to(device)

pretrained = True
if pretrained:
    checkpoint = torch.load(utils.PATH + model_name + ".tar", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    (tr_loss_path_old, val_loss_path_old) = checkpoint['loss']
    batch_size = checkpoint['batch_size']
    validation_split = checkpoint['validation_split']
    total_step_old = checkpoint['total_step']
    n_iteration = checkpoint['n_iteration']
    num_epochs_old = checkpoint['epoch']

train_loader, validation_loader = dataset.loader(batch_size=batch_size,
                                                 validation_split=validation_split,
                                                 shuffle_dataset=True,
                                                 random_seed=49)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
n_iteration = 10
tr_loss_path = np.zeros(shape=(num_epochs, int(total_step/n_iteration)))
val_loss_path = np.zeros(shape=(num_epochs, int(total_step/n_iteration)))

temp_tr_loss = np.zeros(shape=n_iteration)


for epoch in range(num_epochs):
    n = 0
    k = 0
    for i, (tr_images, tr_labels) in enumerate(train_loader):
        tr_images = tr_images.to(device)
        tr_labels = tr_labels.to(device)

        # Forward pass
        tr_outputs = model(tr_images)
        tr_loss = criterion(tr_outputs, tr_labels)
        temp_tr_loss[k] = tr_loss.item()
        # Backward and optimize
        optimizer.zero_grad()
        tr_loss.backward()
        optimizer.step()

        k = k + 1
        if (i+1) % n_iteration == 0:
            k = 0
            temp_val_loss = np.zeros(shape=(len(validation_loader)))
            model.eval()
            with torch.no_grad():
                for j, (val_images, val_labels) in enumerate(validation_loader):
                    val_images = val_images.to(device)
                    val_labels = val_labels.to(device)

                    # Forward pass
                    val_outputs = model(val_images)
                    val_loss = criterion(val_outputs, val_labels)
                    temp_val_loss[j] = val_loss.item()
            model.train()

            tr_loss_path[epoch][n] = temp_tr_loss.mean()
            val_loss_path[epoch][n] = temp_val_loss.mean()
            print('Epoch [{}/{}], Step [{}/{}], Train loss: {:.4f}, Validation loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step,
                          tr_loss_path[epoch][n],val_loss_path[epoch][n]))
            n = n + 1
        if (i+1) % 100 == 0:
            torch.cuda.empty_cache()

if pretrained:
    tr_loss_path = np.vstack([tr_loss_path_old, tr_loss_path])
    val_loss_path = np.vstack([val_loss_path_old, val_loss_path])
    num_epochs = num_epochs_old + num_epochs + 1

torch.save({
            'model_name': model_name,
            'epoch': epoch,
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'validation_split': validation_split,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': (tr_loss_path, val_loss_path),
            'n_iteration': n_iteration,
            'total_step': total_step
            }, utils.PATH + model_name + ".tar")


print("")
