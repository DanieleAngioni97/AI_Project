import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
from pedestrian_dataset import PedestrianDataset

input_size = 28 * 28    # 784
num_classes = 2
batch_size = 64

# set CPU or GPU, if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

dataset = PedestrianDataset(csv_file='DaimlerBenchmark/pedestrian_dataset.csv',
                                    root_dir='./',
                                    transform=transforms.ToTensor())


# Data loader (input pipeline)
train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                           batch_size=batch_size,
                                           shuffle=True)