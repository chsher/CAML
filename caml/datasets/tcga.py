import os
import sys
from os.path import dirname, realpath
sys.path.append(dirname(realpath(__file__)))
from caml.datasets import data_utils

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, set_image_backend, get_image_backend

import accimage
from PIL import Image

JPEG_DIR = '/home/schao/jpeg'
set_image_backend('accimage')

#################### TCGA DATASET ####################
class TCGAdataset(Dataset):
    def __init__(self, df, transform=None, min_tiles=1, num_tiles=100, cancers=None, label='WGD', unit='tile', mag='10.0', H=256, W=256, apply_filter=True, random_seed=31321):
        '''
        df (pandas.DataFrame) : table with metadata (n_tiles, Type, n_tiles_start, n_tiles_end, basename)
        transform (torchvision.transforms.Compose) : pytorch tensor transformations
        min_tiles (int) : minimum number of tiles for patient to be included during sampling
        num_tiles (int) : number of tiles to keep (tile) or sample (slide) per patient
        cancers (list or None) : cancers to include in the dataset; if None, include all
        label (str) : df column name for label annotation
        unit (str) : tile-level or slide-level inputs
        mag (str) : magnification level of the slides
        H (int) : tile height
        W (int) : tile width
        '''
        
        self.df = df
        self.transform = transform
        self.min_tiles = min_tiles
        self.num_tiles = num_tiles
        self.cancers = cancers
        self.label = label
        self.unit = unit
        self.mag = mag
        self.H = H
        self.W = W
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        if apply_filter:
            self.df = data_utils.filter_df(self.df, self.min_tiles, self.cancers)
            
        if self.unit == 'tile':
            self.df['n_tiles'] = self.df['n_tiles'].apply(lambda x: min(x, self.num_tiles))
            self.n_idxs = int(self.df['n_tiles'].sum())
        elif self.unit == 'slide':
            self.n_idxs = self.df.shape[0]
        
    def __len__(self):
        return self.n_idxs

    def __getitem__(self, idx):
        idx = int(idx % self.n_idxs)
        
        if self.unit == 'tile':
            df_idx = np.where((idx < self.df['n_tiles_end']) & (idx >= self.df['n_tiles_start']))[0][0]
        elif self.unit == 'slide':
            df_idx = idx
            
        cancer = self.df.loc[df_idx, 'Type']
        basename = self.df.loc[df_idx, 'basename']
        filepath = os.path.join(JPEG_DIR, cancer, basename + '_files', self.mag)
        jpegs = os.listdir(filepath)
        
        if self.unit == 'tile':
            np.random.shuffle(jpegs)
            jpeg_idx = int(idx - self.df.loc[df_idx, 'n_tiles_start'])
            filenames = np.array([jpegs[jpeg_idx]])
        elif self.unit == 'slide': 
            if len(jpegs) < self.num_tiles:
                filenames = np.random.choice(jpegs, self.num_tiles, replace=True)
            else:
                filenames = np.random.choice(jpegs, self.num_tiles, replace=False)

        if len(filenames) == 1:
            img = data_utils.default_loader(os.path.join(filepath, filenames[0]))
            x = data_utils.process_img(img, self.transform, self.H, self.W)
        elif len(filenames) > 1:
            imgs = []
            for filename in filenames:
                img = data_utils.default_loader(os.path.join(filepath, filename))
                img = data_utils.process_img(img, self.transform, self.H, self.W)
                imgs.append(img)
            x = torch.stack(imgs)
            
        y = torch.FloatTensor([self.df.loc[df_idx, self.label].item()])
        
        return x, y