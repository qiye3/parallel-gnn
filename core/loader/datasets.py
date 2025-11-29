# core/loader/datasets.py

import numpy as np
import pandas as pd
import torch


class InteractionDataset(torch.utils.data.Dataset):
    def __init__(self, parquet_path, num_items):
        df = pd.read_parquet(parquet_path)
        self.users = df["uid"].values
        self.items = df["iid"].values
        self.num_items = int(num_items)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        u = int(self.users[idx])
        pos = int(self.items[idx])
        neg = np.random.randint(0, self.num_items)
        return u, pos, neg
