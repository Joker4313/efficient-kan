import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd


class HARDataSet(Dataset):
    def __init__(self, X_file, y_file):
        X = pd.read_csv(X_file, header=None, sep=r"\s+").to_numpy()
        y = pd.read_csv(y_file, header=None, sep=r"\s+").to_numpy().squeeze() - 1

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class IrisDataSet(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
