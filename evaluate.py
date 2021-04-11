import torch
import torchvision.transforms as transforms
import utils
from model import ConvNet, ConvNet0
from pedestrian_dataset import PedestrianDataset
import matplotlib.pyplot as plt
import numpy as np

# model_name = 'ConvNet_with_augmentation'
model_name = 'ConvNet_without_augmentation'

utils.set_seed(random_seed=49)

# set CPU or GPU, if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = ConvNet().to(device)  # instantiate the model

transform = transforms.ToTensor()

# create the dataset that contains the test set and apply the transformation
dataset = PedestrianDataset(train=False,
                            transform=transform)

# divide the test set in batch
test_loader = dataset.loader(batch_size=64,
                             shuffle_dataset=False)

# load the weight and the other data for the model from the file
checkpoint = torch.load(utils.PATH + model_name + ".tar", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
(tr_loss_path, val_loss_path) = checkpoint['loss']
total_step = checkpoint['total_step']
n_iteration = checkpoint['n_iteration']
num_epochs = checkpoint['num_epochs']
model_name = checkpoint['model_name']

# Test the model
# eval mode (batchnorm uses moving mean/var instead of mini-batch mean/var)
model.eval()
with torch.no_grad():  # for the test is not necessary to calculate the gradients
    correct = 0
    total = 0
    for i, (images, labels) in enumerate(test_loader):
        # pass the current image and the true label to the device
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs[-1].data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if (i + 1) % 1 == 0:
            print('Step [{}/{}], Running accuracy: {:.4f}'
                  .format(i + 1, len(test_loader), correct/total))

    print('Test Accuracy of the model on the 10000 test images: {} %'
          .format(100.0 * correct / total))

# plot the losses of the train and the validation
vector_iterations = np.arange(1, tr_loss_path.ravel().shape[0]+1)*n_iteration
plt.figure()
plt.plot(vector_iterations, tr_loss_path.ravel(), color='blue', label='Train loss')
plt.plot(vector_iterations, val_loss_path.ravel(), color='red', label='Validation loss')
plt.title('Loss of {}'.format(model_name))
plt.legend()
plt.xlabel("iteration")
plt.savefig("images/" + "loss" + model_name + ".png")

plt.show()
