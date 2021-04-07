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


batch_size = 64
validation_split = 0.2

n_iteration = 5

# Hyper-parameters
num_epochs_new = 10
learning_rate = 0.001

model_name1 = "ConvNet1_with_augmentation"
model_name2 = "ConvNet1_without_augmentation"
model_names_list = [model_name1, model_name2]
save = True
pretrained = False


transform1 = transforms.ToTensor()
transform2 = transforms.Compose([
        transforms.ColorJitter(brightness=(0.8, 1.2),
                               contrast=(0.8, 1.2),
                               saturation=(0.8, 1.2)),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.6)),
        transforms.ToTensor(),
        ])
transform_list = [transform1, transform2]

# set CPU or GPU, if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model_orig = ConvNet1().to(device)

for transform, model_name in list(zip(transform_list, model_names_list)):
    utils.set_seed(random_seed=49)

    dataset = PedestrianDataset(train=True,
                                transform=transform)

    model = ConvNet1().to(device)
    model.load_state_dict(model_orig.state_dict())

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
                                                     shuffle_dataset=True)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    total_step = len(train_loader)
    tr_loss_path = np.zeros(shape=(num_epochs_new, int(total_step/n_iteration)))
    val_loss_path = np.zeros(shape=(num_epochs_new, int(total_step/n_iteration)))

    temp_tr_loss = np.zeros(shape=n_iteration)

    if pretrained:
        tr_loss_path = np.vstack([tr_loss_path_old, tr_loss_path])
        val_loss_path = np.vstack([val_loss_path_old, val_loss_path])
        num_epochs = num_epochs_old + num_epochs_new + 1
    else:
        num_epochs = num_epochs_new

    for epoch in range(num_epochs_new):
        if pretrained:
            epoch = num_epochs_old + 1 + epoch
        n = 0   # n è il numero di iterazioni ogni n_iteration (esattamente la stessa misura del grafico)
        k = 0   # è un indice temporaneo che va da 0 a n_iteration e serve per riempire temp_tr_loss
        # i è l' indice dei batch
        for i, (tr_images, tr_labels) in enumerate(train_loader):
            tr_images = tr_images.to(device)
            tr_labels = tr_labels.to(device)

            # Forward pass
            tr_outputs = model(tr_images)[-1]
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
                        val_outputs = model(val_images)[-1]
                        val_loss = criterion(val_outputs, val_labels)
                        temp_val_loss[j] = val_loss.item()
                model.train()
                tot_val_loss = temp_val_loss.mean()
                tr_loss_path[epoch][n] = temp_tr_loss.mean()
                val_loss_path[epoch][n] = tot_val_loss
                if n == 0:
                    min_val_loss = tot_val_loss
                if tot_val_loss < min_val_loss:
                    min_val_loss = tot_val_loss

                    if save:
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

                print('Epoch [{}/{}], Step [{}/{}], Train loss: {:.4f}, Validation loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step,
                              tr_loss_path[epoch][n], val_loss_path[epoch][n]))
                n = n + 1
            if (i+1) % 100 == 0:
                torch.cuda.empty_cache()

print("")
