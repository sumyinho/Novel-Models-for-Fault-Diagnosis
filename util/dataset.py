import torch
from torch.utils.data import Dataset
import numpy as np


class VibrationSignalDataset(Dataset):

    def __init__(self, data, labels, channel_last=False, transform=None, **kwargs):
        self.data = torch.FloatTensor(np.expand_dims(data, axis=1))
        self.labels = torch.LongTensor(labels)

        if transform is not None:
            print(self.data.shape)
            self.data = transform(self.data, **kwargs)
        print("data:", self.data.shape)

        if channel_last is True:
            self.data = self.data.transpose(-1, -2)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]


class TimeFrequencyDataset(Dataset):

    def __init__(self, data, labels, transform=None, **kwargs):
        self.data = data.unsqueeze(dim=1)
        self.labels = labels.type(torch.int64).squeeze()

        if transform is not None:
            self.data = transform(self.data, **kwargs)
        print("data:", self.data.shape)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]
