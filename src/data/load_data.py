import torch
from torch.utils.data import Dataset
import pandas as pd


class HARDataSet(Dataset):
    def __init__(self, X_file, y_file):
        self.X = (
            pd.read_csv(X_file, header=None, sep=r"\s+").to_numpy().astype("float32")
        )
        self.y = (
            pd.read_csv(y_file, header=None, sep=r"\s+")
            .to_numpy()
            .squeeze()
            .astype("long")
            - 1
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(
            self.y[idx], dtype=torch.long
        )
