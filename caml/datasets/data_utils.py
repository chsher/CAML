import os
import sys
from os.path import dirname, realpath
sys.path.append(dirname(realpath(__file__)))
from caml.datasets import tcga

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

#################### IMAGE TRANSFORMING ####################
normalize = transforms.Normalize(mean=[0.7156, 0.5272, 0.6674], std=[0.2002, 0.2330, 0.1982]) # ['BLCA', 'BRCA', 'COAD', 'HNSC', 'LUAD', 'LUSC', 'READ', 'STAD']
transform = transforms.Compose([transforms.ToPILImage(),
                                transforms.RandomVerticalFlip(),
                                transforms.RandomHorizontalFlip(),
                                transforms.ColorJitter(hue=0.02, saturation=0.1),
                                transforms.ToTensor(), normalize])

#################### DATA SPLITTING ####################
def split_datasets_by_sample(df, train_frac=0.8, val_frac=0.2, random_seed=31321, 
                             transform=None, min_tiles=1, num_tiles=100, cancers=None, label='WGD', unit='tile', mag='10.0', H=256, W=256):
    '''
    - currently only handles TCGAdataset datasets
    '''
    if random_seed is not None:
        np.random.seed(random_seed)
        
    df = filter_df(df, min_tiles, cancers)
        
    idxs = np.arange(df.shape[0])
    np.random.shuffle(idxs)
    
    val_start = int(train_frac * len(idxs))
    dfs = [filter_df(df, idxs=idxs[:val_start])]

    if train_frac + val_frac != 1:
        test_start = int((train_frac + val_frac) * len(idxs))
        dfs.append(filter_df(df, idxs=idxs[val_start:test_start]))
        dfs.append(filter_df(df, idxs=idxs[test_start:]))
    else:
        dfs.append(filter_df(df, idxs=idxs[val_start:]))

    dss = []
    for i, d in enumerate(dfs):
        if i == 0:
            ds = tcga.TCGAdataset(d, transform, min_tiles, num_tiles, cancers, label, unit, mag, H, W, apply_filter=False) 
        else:
            ds = tcga.TCGAdataset(d, transforms.Compose([normalize]), min_tiles, num_tiles, cancers, label, unit, mag, H, W, apply_filter=False) 
        dss.append(ds)
        
    return dss

#################### DATA FILTERING ####################
def filter_df(df, min_tiles=None, cancers=None, idxs=None):
    if idxs is not None:
        df = df.loc[idxs, :]
        
    if min_tiles is not None:
        df = df[df['n_tiles'] >= min_tiles]
        
    if cancers is not None:
        df = df[df['Type'].isin(cancers)]
    
    df.reset_index(drop=True, inplace=True)
    
    df['n_tiles_end'] = df['n_tiles'].cumsum()
    df['n_tiles_start'] = np.concatenate((np.array([0]), df['n_tiles_end'].values[:-1]))
    
    return df

#################### STATS COMPUTING ####################
def compute_stats(train_dataset, batch_size=50, n_batches=100):
    loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=True)
    
    tracker = []
    for i, (x, y) in tqdm(enumerate(loader)):
        if i >= n_batches:
            break
        else:
            tracker.append(torch.mean(x, dim=[0, 2, 3]))
            
    means = torch.stack(tracker)
    mu = torch.mean(means, dim=0)
    
    tracker2 = []
    for i, (x, y) in tqdm(enumerate(loader)):
        if i >= n_batches:
            break
        else:
            t = torch.pow(x.transpose(1, -1) - mu, 2) # swapped dims 1 and 3 to enable broadcasting
            tracker2.append(torch.mean(t, dim=[0, 1, 2]))
            
    variances = torch.stack(tracker2)
    var = torch.mean(variances, dim=0)
    sig = torch.sqrt(var)
    
    return mu, sig

#################### IMAGE LOADING ####################
def accimage_loader(filepath):
    try:
        return accimage.Image(filepath)
    except IOError:
        return pil_loader(filepath)

def pil_loader(filepath):
    with open(filepath, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def default_loader(filepath):
    if get_image_backend() == 'accimage':
        return accimage_loader(filepath)
    else:
        return pil_loader(filepath)

#################### IMAGE PRE-PROCESSING ####################
def img_to_tensor(img):
    '''
    convert image to float32 tensor
    '''
    
    if isinstance(img, accimage.Image):
        img_np = np.zeros([img.channels, img.height, img.width], dtype=np.float32)
        img.copyto(img_np)
        img = torch.from_numpy(img_np)
    elif isinstance(img, Image.Image):
        transform_pil = transforms.Compose([transforms.ToTensor()])
        img = transform_pil(img)
        
    return img

def pad_img(img, H, W):
    '''
    - pad image to height H and width W
    - padding starts from last dimension
    - assumes H x W are the last two dimensions
    '''
    
    gap_H = (H - img.shape[1])
    gap_W = (W - img.shape[2])
    padding = [gap_W // 2, gap_W // 2 + gap_W % 2, gap_H // 2, gap_H // 2 + gap_H % 2]
    
    return F.pad(img, padding, mode='constant', value=0)     

def process_img(img, transform, H, W):
    img = img_to_tensor(img)
    
    if transform is not None:
        img = transform(img)
        
    img = pad_img(img, H, W)
    
    return img