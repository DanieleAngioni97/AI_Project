"""
Show images of the deep space representation of samples from the test set
"""

import torch
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import utils
from model import ConvNet1
from pedestrian_dataset import PedestrianDataset
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

def imshow(img):
    img = (img)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

n_images = 4
layer = 3


if layer is None:
    min, max = 0, 5
else:
    min, max = layer, layer+1


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
utils.set_seed(49)

model = ConvNet1().to(device)

dataset = PedestrianDataset(train=False,
                            transform=transforms.ToTensor())

test_loader = dataset.loader(batch_size=n_images,
                             shuffle_dataset=True)

checkpoint = torch.load(utils.PATH + "ConvNet1_without_augmentation.tar", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

test_iterator = iter(test_loader)

(images, labels) = test_iterator.next()
(images, labels) = test_iterator.next()

model.eval()

classes = ['non-pedestrian', 'pedestrian']
with torch.no_grad():
    for (image, label) in list(zip(images, labels)):
        outputs = model(image.unsqueeze(dim=0))
        _, y_pred = torch.max(outputs[-1].data, 1)

        image = image.numpy().transpose((1, 2, 0))
        plt.imshow(image)
        plt.title('True label : {}\nPredicted : {}'.format(classes[label.item()],
                                                           classes[y_pred.item()]))
        plt.axis('off')
        plt.show()

        nrows = [8, 8, 16, 16, 16, 8]

        for i in range(min, max):
            if i == 4:
                plot_tensor = outputs[i].view(16, 16).unsqueeze(0).unsqueeze(0)
            else:
                plot_tensor = outputs[i].unsqueeze(dim=2)[0]
            fig = make_grid(plot_tensor,
                            normalize=True,
                            nrow=nrows[i],
                            pad_value=0
                            )
            plt.title('Output of {}° conv layer, label : {}, \nPrediction: {}, confidence: {:.5f}'
                      .format(i + 1,
                              classes[label.item()],
                              classes[y_pred.item()],
                              F.softmax(outputs[5][0]).tolist()[label.item()]))
            plt.axis('off')
            imshow(fig)

    # for (image, label) in test_loader:
    #     outputs = model(image)
    #     _, y_pred = torch.max(outputs[-1].data, 1)
    #
    #     image = image.numpy()[0].transpose((1, 2, 0))
    #     plt.imshow(image)
    #     plt.title('True label : {}\nPredicted : {}'.format(classes[label.item()],
    #                                                        classes[y_pred.item()]))
    #     plt.axis('off')
    #     plt.show()
    #
    #     nrows = [8, 8, 16, 16]
    #     i = 3
    #     # for i in range(0, 4):
    #     fig = make_grid(outputs[i].unsqueeze(dim=2)[0],
    #                     normalize=True,
    #                     nrow=nrows[i],
    #                     pad_value=0
    #                     )
    #     plt.title('Output of {}° conv layer, label : {}, \nPrediction: {}, confidence: {:.5f}'
    #               .format(i+1,
    #                       classes[label.item()],
    #                       classes[y_pred.item()],
    #                       F.softmax(outputs[5][0]).tolist()[label.item()]))
    #     plt.axis('off')
    #     imshow(fig)
    #     print("")
    #








