import torch
import torchvision.transforms as transforms
import utils
from model import ConvNet0
from pedestrian_dataset import PedestrianDataset

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = ConvNet0().to(device)
optimizer = torch.optim.Adam(model.parameters())

model = ConvNet0().to(device)

dataset = PedestrianDataset(train=False,
                            transform=transforms.ToTensor())

test_loader = dataset.loader(batch_size=1024,
                             shuffle_dataset=False,
                             random_seed=0)

checkpoint = torch.load(utils.PATH + "modello_swag.tar", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
# (tr_loss_path, val_loss_path) = checkpoint['loss']

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
