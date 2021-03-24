import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from model import ConvNet
from pedestrian_dataset import PedestrianDataset
import utils

# INPUT PARAMETERS

# Hyper-parameters
num_epochs = 1
learning_rate = 0.001
pretrained = False

# set CPU or GPU, if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(0)

dataset = PedestrianDataset(train=True,
                            transform=transforms.ToTensor())

train_loader, validation_loader = dataset.loader(batch_size=128,
                                                 validation_split=.1,
                                                 shuffle_dataset=True,
                                                 random_seed=0)

model = ConvNet().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

if pretrained:
    checkpoint = torch.load(utils.PATH + "modello_swag.tar", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']


# Train the model
total_step = len(train_loader)
tr_loss_path = np.zeros(shape=(num_epochs, total_step))
val_loss_path = np.zeros(shape=(num_epochs, total_step))

model.train()
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        tr_loss_path[epoch][i] = loss.item()

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # todo: aggiungere la loss sul validation, salvare le loss ecc

        if (i+1) % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
        if (i+1) % 100 == 0:
            torch.cuda.empty_cache()

torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
            }, utils.PATH + "modello_swag.tar")

plt.figure()
plt.plot(tr_loss_path.ravel())
plt.title('Loss')
plt.xlabel("iteration")
plt.show()

print("")
