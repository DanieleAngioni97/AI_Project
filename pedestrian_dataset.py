from __future__ import print_function, division
import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import numpy as np
import utils
import random


from read_pgm import read_pgm_reshape


class PedestrianDataset(Dataset):
    """Pedestrian dataset."""

    def __init__(self, train=True, root_dir='./', transform=None, shape=(32, 32)):
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
        self.shape = shape
        self.train = train

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
        sample = (image, label)

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

        if self.train:
            train_size = int(np.floor(dataset_size * (1 - validation_split)))
            validation_size = int(dataset_size - train_size)
            assert train_size + validation_size == dataset_size


            if shuffle_dataset:
                np.random.shuffle(indices)

            train_indices = indices[:train_size]
            val_indices = indices[train_size:]

            # torch.manual_seed(random_seed)
            # Creating PT data samplers and loaders:
            train_sampler = SubsetRandomSampler(train_indices)
            valid_sampler = SubsetRandomSampler(val_indices)

            train_loader = torch.utils.data.DataLoader(self, batch_size=batch_size, sampler=train_sampler)
            validation_loader = torch.utils.data.DataLoader(self, batch_size=batch_size, sampler=valid_sampler)

            return train_loader, validation_loader
        else:
            test_sampler = SubsetRandomSampler(indices)
            test_loader = torch.utils.data.DataLoader(self, batch_size=batch_size)

        return test_loader


if __name__ == "__main__":
    import numpy as np
    import torchvision.transforms as transforms
    from torch.utils.data import SubsetRandomSampler
    from pedestrian_dataset import PedestrianDataset

    dataset = PedestrianDataset(csv_file='DaimlerBenchmark/pedestrian_dataset.csv',
                                root_dir='./',
                                transform=transforms.ToTensor())

    tr_loader, _, _ = dataset.loader(batch_size=128,
                                        train_split=.8,
                                        validation_split=.2,
                                        shuffle_dataset=True,
                                        random_seed=0)
    _, val_loader, _ = dataset.loader(batch_size=128,
                                     train_split=.8,
                                     validation_split=.2,
                                     shuffle_dataset=True,
                                     random_seed=0)
    _, _, ts_loader = dataset.loader(batch_size=1024,
                                     train_split=.8,
                                     validation_split=.2,
                                     shuffle_dataset=True,
                                     random_seed=0)

    print("")
