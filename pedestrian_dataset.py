from __future__ import print_function, division
import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from PIL import Image
import numpy as np
import utils
import random

from read_pgm import read_pgm_reshape


class PedestrianDataset(Dataset):
    """Pedestrian dataset."""

    def __init__(self, train=True, root_dir='./', transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        csv_path = os.path.join(utils.ROOT_DIR_DATASET, utils.TRAIN_PATH if train else utils.TEST_PATH)
        self.pedestrian_frame = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.pedestrian_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):    # Returns True if idx is a PyTorch tensor
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.pedestrian_frame.iloc[idx, 0])
        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)

        label = self.pedestrian_frame.iloc[idx, 1]
        label = np.array(label)
        sample = (image[:3], label)

        return sample

    def loader(self, batch_size=64,
               validation_split=.2,
               shuffle_dataset=False,
               random_seed=0):

        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        random.seed(random_seed)

        # Creating data indices for training and validation splits:
        dataset_size = len(self)
        indices = list(range(dataset_size))

        if shuffle_dataset:
            for i in range(10):
                np.random.shuffle(indices)

        if self.train:
            train_size = int(np.floor(dataset_size * (1 - validation_split)))
            validation_size = int(dataset_size - train_size)
            assert train_size + validation_size == dataset_size

            train_indices = indices[:train_size]
            val_indices = indices[train_size:]

            # Creating PT data samplers and loaders:
            #train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
            #valid_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
            train_sampler = torch.utils.data.Subset(self, train_indices)
            valid_sampler = torch.utils.data.Subset(self, val_indices)
            #train_sampler = torch.utils.data.SequentialSampler(self)
            #valid_sampler = torch.utils.data.SequentialSampler(self)

            train_loader = torch.utils.data.DataLoader(train_sampler, batch_size=batch_size)
            validation_loader = torch.utils.data.DataLoader(valid_sampler, batch_size=batch_size)

            return train_loader, validation_loader
        else:
            #valutare la possibilità dello shuffle per il test
            test_loader = torch.utils.data.DataLoader(self, batch_size=batch_size)

        return test_loader


if __name__ == "__main__":
    import numpy as np
    import torchvision.transforms as transforms
    from pedestrian_dataset import PedestrianDataset

    batch_size = 128
    validation_split = 0.2

    dataset_train_val = PedestrianDataset(train=True,
                                          transform=transforms.ToTensor())

    train_loader, validation_loader = dataset_train_val.loader(batch_size=batch_size,
                                                               validation_split=validation_split,
                                                               shuffle_dataset=True,
                                                               random_seed=49)

    dataset_test = PedestrianDataset(train=False,
                                     transform=transforms.ToTensor())

    test_loader = dataset_test.loader(batch_size=batch_size,
                                     shuffle_dataset=False,
                                     random_seed=49)

    # img, label = iter(train_loader).next()

    for img, label in train_loader:
        pass

    print(dataset_train_val.cnt)

    print("")

