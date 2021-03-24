import torch
import torchvision.transforms as transforms
import utils
from model import ConvNet
from pedestrian_dataset import PedestrianDataset

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = ConvNet().to(device)

dataset = PedestrianDataset(csv_file='DaimlerBenchmark/pedestrian_dataset.csv',
                            root_dir='./',
                            transform=transforms.ToTensor())

_, _, test_loader = dataset.loader(batch_size=64,
                                   train_split=.8,
                                   validation_split=.2,
                                   shuffle_dataset=True,
                                   random_seed=0)

checkpoint = torch.load(utils.PATH + "modello_swag.tar", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

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
