import torch
import torchvision.transforms as transforms
import utils
from model import ConvNet1
from pedestrian_dataset import PedestrianDataset
import matplotlib.pyplot as plt
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = ConvNet1().to(device)

dataset = PedestrianDataset(train=False,
                            transform=transforms.ToTensor())

test_loader = dataset.loader(batch_size=16,
                             shuffle_dataset=False,
                             random_seed=0)

checkpoint = torch.load(utils.PATH + "modello_too_swag.tar", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

(images, labels) = iter(test_loader).next()

model.eval()
with torch.no_grad():
    outputs = model(images)[-1]
    _, y_preds = torch.max(outputs.data, 1)

label_str = ['non-pedestrian', 'pedestrian']

for (image, label, y_pred) in list(zip(images, labels, y_preds)):
    image = np.transpose(image.numpy(), (2, 3, 1))
    plt.imshow(image, plt.cm.gray)
    plt.title('True label : {}\nPredicted : {}'.format(label_str[label.item()],
                                                     label_str[y_pred.item()]))
    plt.show()

print("")






