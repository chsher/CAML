import os
import sys
from os.path import dirname, realpath
sys.path.append(dirname(realpath(__file__)))
from caml.datasets import tcga

from torch.utils.data import Dataset

class MAMLdataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets
        self.n_idxs = min(len(d) for d in self.datasets)
        
    def __len__(self):
        return self.n_idxs
    
    def __getitem__(self, idx):
        idx = int(idx % self.n_idxs)

        xs = []
        ys = []

        for d in self.datasets:
            x, y in d.__getitem__(idx)
            xs.append(x)
            ys.append(y)

        xs = torch.stack(xs)
        ys = torch.cat(ys)

        return xs, ys