from torch.utils.data import DataLoader
from torchvision import datasets
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from prefetch_generator import BackgroundGenerator


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class Setloader(Dataset):
    def __init__(self, data, label):
        super(Setloader, self).__init__()
        self.data = data
        self.label = label

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return self.data.shape[0]


class TestSetloader(Dataset):
    def __init__(self, data):
        super(TestSetloader, self).__init__()
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.data.shape[0]