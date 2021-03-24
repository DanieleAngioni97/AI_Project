from __future__ import print_function, division
import os
import torch
import pandas as pd
# from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms, utils

from prova_lettura_immagini import read_pgm_reshape


class PedestrianDataset(Dataset):
    """Pedestrian dataset."""

    def __init__(self, csv_file, root_dir, transform=None, shape=(32, 32)):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.pedestrian_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.shape = shape

    def __len__(self):
        return len(self.pedestrian_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):    # Returns True if idx is a PyTorch tensor
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.pedestrian_frame.iloc[idx, 0])
        # dataframe.iloc[start_row:end_row,start_col:end_col]
        # image = io.imread(img_name) #Load an image from file.
        image = read_pgm_reshape(img_name, self.shape)

        if self.transform:
            image = self.transform(image)

        label = self.pedestrian_frame.iloc[idx, 1]
        label = np.array(label)
        # sample = {'image': image, 'label': label}
        sample = (image, label)

        return sample

    def loader(self, batch_size=64,
               train_split=.8,
               validation_split=.2,
               shuffle_dataset=True,
               random_seed=0):

        # Creating data indices for training and validation splits:
        dataset_size = len(self)  # 49000

        train_size = int(np.floor(train_split * dataset_size))
        print("Train size + Val size = ", train_size)
        test_size = int(dataset_size - train_size)
        train_size = int(np.floor(train_size * (1 - validation_split)))
        validation_size = int(dataset_size - train_size - test_size)

        print("train_size = ", train_size)
        print("validation_size = ", validation_size)
        print("test_size", test_size)
        print("Sum = ", train_size + validation_size + test_size)

        assert train_size + validation_size + test_size == dataset_size

        # Train size + Val size =  39200
        # train_size =  31360
        # validation_size =  7840
        # test_size 9800

        indices = list(range(dataset_size))
        if shuffle_dataset:
            np.random.seed(random_seed)
            np.random.shuffle(indices)

        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + validation_size]
        test_indices = indices[train_size + validation_size:]

        # Creating PT data samplers and loaders:
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)
        test_sampler = SubsetRandomSampler(test_indices)

        train_loader = torch.utils.data.DataLoader(self, batch_size=batch_size, sampler=train_sampler)
        validation_loader = torch.utils.data.DataLoader(self, batch_size=batch_size, sampler=valid_sampler)
        test_loader = torch.utils.data.DataLoader(self, batch_size=batch_size, sampler=test_sampler)

        return train_loader, validation_loader, test_loader
