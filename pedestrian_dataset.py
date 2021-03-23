from __future__ import print_function, division
import os
import torch
import pandas as pd
# from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
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
        if torch.is_tensor(idx): # Returns True if idx is a PyTorch tensor
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.pedestrian_frame.iloc[idx, 0])
        # dataframe.iloc[start_row:end_row,start_col:end_col]
        # image = io.imread(img_name) #Load an image from file.
        image = read_pgm_reshape(img_name, self.shape)
        label = self.pedestrian_frame.iloc[idx, 1]
        label = np.array([label])
        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample