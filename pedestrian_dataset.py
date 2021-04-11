from __future__ import print_function, division
import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import utils


class PedestrianDataset(Dataset):
    """Pedestrian dataset."""

    def __init__(self, train=True, root_dir='./', transform=None):
        """

        :param train: boolean, set True if train set is required, otherwise set False to pick the test set
        :param root_dir: directory in wich the dataset is stored
        :param transform: transform to apply every time a sample is picked
        """
        csv_path = os.path.join(utils.ROOT_DIR_DATASET, utils.TRAIN_CSV_FNAME if train else utils.TEST_CSV_FNAME)
        self.pedestrian_frame = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.pedestrian_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):    # Returns True if idx is a PyTorch tensor
            idx = idx.tolist()

        # each row correspond to a sample in the train/test set (depending on self.train)
        # for each sample (row), the first column is the path from content root of the file image
        # the second column is its associated ground truth (0 if non-pedestrian, 1 if pedestrian)

        img_fname = os.path.join(self.root_dir,
                                 self.pedestrian_frame.iloc[idx, 0])

        image = Image.open(img_fname)

        if self.transform:
            image = self.transform(image)

        label = self.pedestrian_frame.iloc[idx, 1]
        label = np.array(label)

        # create a tuple containing the sample and its label
        # slice the image tensor to exclude fourth channel if present
        sample = (image[:3], label)

        return sample

    def loader(self, batch_size=64,
               validation_split=.2,
               shuffle_dataset=False):
        """

        :param batch_size: int, number of image selected for each iteration
        :param validation_split: fraction of train set selected as validation set
        :param shuffle_dataset: boolean, if True the dataset is shuffled
        :return: tuple with (trainloader, validationloader) if self.train is True, testloader otherwise
        """

        # Creating data indices:
        dataset_size = len(self)
        indices = list(range(dataset_size))

        if shuffle_dataset:
            # shuffle more than one time to be sure that the dataset is shuffled appropriately
            for i in range(20):
                np.random.shuffle(indices)

        # if self.train is True, a validation set is created from the train set
        if self.train:
            train_size = int(np.floor(dataset_size * (1 - validation_split)))
            validation_size = int(dataset_size - train_size)
            assert train_size + validation_size == dataset_size

            train_indices = indices[:train_size]
            val_indices = indices[train_size:]

            # create subset of the dataset with selected indices
            train_sampler = torch.utils.data.Subset(self, train_indices)
            valid_sampler = torch.utils.data.Subset(self, val_indices)

            train_loader = torch.utils.data.DataLoader(train_sampler, batch_size=batch_size)
            validation_loader = torch.utils.data.DataLoader(valid_sampler, batch_size=batch_size)

            return train_loader, validation_loader

        # otherwise return the shuffled test set
        else:
            test_sampler = torch.utils.data.Subset(self, indices)
            test_loader = torch.utils.data.DataLoader(test_sampler, batch_size=batch_size)

        return test_loader


if __name__ == "__main__":
    import torchvision.transforms as transforms
    from pedestrian_dataset import PedestrianDataset

    batch_size = 128
    validation_split = 0.2

    dataset_train_val = PedestrianDataset(train=True,
                                          transform=transforms.ToTensor())

    train_loader, validation_loader = dataset_train_val.loader(batch_size=batch_size,
                                                               validation_split=validation_split,
                                                               shuffle_dataset=True)

    dataset_test = PedestrianDataset(train=False,
                                     transform=transforms.ToTensor())

    test_loader = dataset_test.loader(batch_size=batch_size,
                                     shuffle_dataset=False)


