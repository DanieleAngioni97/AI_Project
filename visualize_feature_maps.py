"""
Show images of the deep space representation of samples from the test set
"""

import torch
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import utils
from utils import imshow
from model import ConvNet
from pedestrian_dataset import PedestrianDataset
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F


def visualize(n_images=1, id_batch=0, layer=None):
    """

    :param n_images: number of images to be visualized
    :param id_batch: select the batch
    :param layer: None if all layer outputs has to be visualized, else specify a layer ID
    """

    # range of layers if layer is None
    min_layer, max_layer = 0, 5

    if layer is not None:   # convenience range for a single layer output visualization
        min_layer, max_layer = layer, layer + 1

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    utils.set_seed(49)

    # instantiate the model
    model = ConvNet().to(device)

    # load the test set
    dataset = PedestrianDataset(train=False,
                                transform=transforms.ToTensor())

    test_loader = dataset.loader(batch_size=n_images,
                                 shuffle_dataset=True)

    # load weight of the pretrained model with higher accuracy
    checkpoint = torch.load(utils.MODELS_PATH + "ConvNet_without_augmentation.tar", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # iterate the test loader to obtain n_images samples
    test_iterator = iter(test_loader)
    for batch in range(0, id_batch + 1):
        (images, labels) = test_iterator.next()

    model.eval()    # evaluation mode

    classes = ['non-pedestrian', 'pedestrian']

    with torch.no_grad():
        for (image, label) in list(zip(images, labels)):
            # pass the image with the dimension for number of images in batches to be
            # processable by the network
            outputs = model(image.unsqueeze(dim=0))

            # compute the prediction
            _, y_pred = torch.max(outputs[-1].data, 1)

            # plot the input image
            image = image.numpy().transpose((1, 2, 0))
            plt.imshow(image)
            plt.title('True label : {}\nPredicted : {}'.format(classes[label.item()],
                                                               classes[y_pred.item()]))
            plt.axis('off')
            plt.show()

            # convenience list to be passed in make_grid
            nrows = [8, 8, 16, 16, 16, 8]

            # loop in the selected range of layers
            for i in range(min_layer, max_layer):
                # if i points at the fourth layer the output of the first fc layer is reshaped to look like an image
                if i == 4:
                    plot_tensor = outputs[i].view(16, 16).unsqueeze(0).unsqueeze(0)
                else:
                    plot_tensor = outputs[i].unsqueeze(dim=2)[0]

                # plot a grid with the output of every filter of the i-th layer
                fig = make_grid(plot_tensor,
                                normalize=True,
                                nrow=nrows[i],
                                pad_value=0
                                )

                plt.title('Output of {}° layer'
                          .format(i + 1))

                imshow(fig)

def stats_feature_map(batch_size = 32, layer=3, train=False):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    utils.set_seed(49)

    # instantiate the model
    model = ConvNet().to(device)

    # load the test set
    dataset = PedestrianDataset(train=train,
                                transform=transforms.ToTensor())
    if train:
        data_loader, _ = dataset.loader(batch_size=batch_size,
                                     shuffle_dataset=True)
    else:
        data_loader = dataset.loader(batch_size=batch_size,
                                     shuffle_dataset=True)

    # load weight of the pretrained model with higher accuracy
    checkpoint = torch.load(utils.MODELS_PATH + "ConvNet_without_augmentation.tar", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    classes = ['non-pedestrian', 'pedestrian']

    # convenience list to be passed in make_grid
    nrows = [8, 8, 16, 16, 16, 8]

    model.eval()  # evaluation mode
    temp_ped = 0
    temp_nonped = 0
    total_ped = 0
    total_nonped = 0
    print('Computing stats...')
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            outputs = model(images)
            if i == 0:
                temp_ped = torch.zeros(size=outputs[layer][0].shape)
                temp_nonped = torch.zeros(size=outputs[layer][0].shape)

            for output, label in list(zip(outputs[layer], labels)):
                if label == 0:
                    total_nonped += 1
                    temp_nonped += output
                else:
                    total_ped += 1
                    temp_ped += output
    temp_nonped /= total_nonped
    temp_ped /= total_ped
    print('Finished.')

    for i, plot_tensor in enumerate([temp_nonped, temp_ped]):
        plot_tensor = plot_tensor.unsqueeze(dim=1)

        # plot a grid with the output of every filter of the i-th layer
        fig = make_grid(plot_tensor,
                        normalize=True,
                        nrow=nrows[layer],
                        pad_value=0
                        )
        plt.title('Mean output of {}° layer, label : {}'.format(layer + 1, classes[i]))
        imshow(fig)



if __name__ == '__main__':
    visualize(n_images=1, id_batch=1, layer=None)
    # stats_feature_map(train=False)
