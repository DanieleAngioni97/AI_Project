import torch
import torchvision.transforms as transforms
import utils
from model import ConvNet1
from pedestrian_dataset import PedestrianDataset
import matplotlib.pyplot as plt
import numpy as np


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = ConvNet1().to(device)

model = ConvNet1().to(device)

dataset = PedestrianDataset(train=False,
                            transform=transforms.ToTensor())

test_loader = dataset.loader(batch_size=1024,
                             shuffle_dataset=False,
                             random_seed=0)

checkpoint = torch.load(utils.PATH + "modello_too_swag.tar", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
(tr_loss_path, val_loss_path) = checkpoint['loss']
total_step = checkpoint['total_step']
n_iteration = checkpoint['n_iteration']
num_epochs = checkpoint['num_epochs']
# Test the model
# eval mode (batchnorm uses moving mean/var instead of mini-batch mean/var)
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for i, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if (i + 1) % 1 == 0:
            print('Step [{}/{}], Running accuracy: {:.4f}'
                  .format(i + 1, len(test_loader), correct/total))

    print('Test Accuracy of the model on the 10000 test images: {} %'
          .format(100.0 * correct / total))


vector_iterations = (np.arange(1, int((total_step/n_iteration))*(num_epochs+1)+1))*n_iteration
plt.figure()
plt.plot(vector_iterations, tr_loss_path.ravel(), color='blue', label='Train loss')
plt.plot(vector_iterations, val_loss_path.ravel(), color='red', label='Validation loss')
# plt.plot(x, y, color='green', marker='o', linestyle='dashed', linewidth=2, markersize=12)
plt.title('Train loss and Validation loss')
plt.legend()
plt.xlabel("iteration")
plt.show()