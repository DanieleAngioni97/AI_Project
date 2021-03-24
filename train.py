import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
from torch.utils.data import SubsetRandomSampler

from model import ConvNet
from pedestrian_dataset import PedestrianDataset
import utils

# set CPU or GPU, if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

dataset = PedestrianDataset(csv_file='DaimlerBenchmark/pedestrian_dataset.csv',
                            root_dir='./',
                            transform=transforms.ToTensor())

train_loader, validation_loader, _ = dataset.loader(batch_size=64,
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

        # todo: aggiungere la loss sul validation, salvare le loss ecc
        k = k + 1
        if (i+1) % n_iteration == 0:
            k = 0
            temp_val_loss = np.zeros(shape=(len(validation_loader)))
            for j, (val_images, val_labels) in enumerate(validation_loader):
                val_images = tr_images.to(device)
                val_labels = tr_labels.to(device)

                # Forward pass
                val_outputs = model(val_images)
                val_loss = criterion(val_outputs, val_labels)
                temp_val_loss[j] = val_loss.item()

            tr_loss_path[epoch][n] = temp_tr_loss.mean()
            val_loss_path[epoch][n] = temp_val_loss.mean()
            print('Epoch [{}/{}], Step [{}/{}], Train loss: {:.4f}, Validation loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step,
                    tr_loss_path[epoch][n],val_loss_path[epoch][n]))
            n = n + 1
        if (i+1) % 100 == 0:
            torch.cuda.empty_cache()

#torch.save({
#            'epoch': epoch,
#            'model_state_dict': model.state_dict(),
#            'optimizer_state_dict': optimizer.state_dict(),
#            'loss': tr_loss
#            }, utils.PATH + "modello_swag.tar")

plt.figure()
plt.plot(tr_loss_path.ravel(), color='blue', label='Train loss')
plt.plot(val_loss_path.ravel(), color='red', label='Validation loss')
#plt.plot(x, y, color='green', marker='o', linestyle='dashed', linewidth=2, markersize=12)
plt.title('Train loss and Validation loss')
plt.legend()
plt.xlabel("iteration")
plt.show()

print("")
