import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

#Basic dataset implementation; included label count for testing purposes
class CogDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.from_numpy(data)
        self.labels = torch.from_numpy(labels)
        self.num_labels = torch.bincount(self.labels)
    
    def __len__(self):
        return len(self.data)
        

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
