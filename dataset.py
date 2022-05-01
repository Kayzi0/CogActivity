import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

#Basic dataset implementation; included label count for testing purposes
class CogDataset(Dataset):
    def __init__(self, data, labels, train=False):
        self.data = torch.from_numpy(data)
        self.labels = torch.from_numpy(labels)
        self.num_labels = torch.bincount(self.labels)
        self.train = train

        # split dataset in train/valid set
        ratio = 0.8
        split = int(len(self.data)*ratio)
        perm = torch.randperm(len(self.data))

        # shape = [datasets, time series, channels, devices]
        self.data_train = [self.data[i] for i in perm[:split]]
        self.labels_train = [self.labels[i] for i in perm[:split]]
        self.data_val = [self.data[i] for i in perm[split:]]
        self.labels_val = [self.labels[i] for i in perm[split:]]

    
    def __len__(self):
        if self.train:
            return len(self.data_train)
        
        return len(self.data_val)
        

    def __getitem__(self, idx):
        if self.train:
            return self.data_train[idx], self.labels_train[idx]
        
        return self.data_val[idx], self.labels_val[idx]

