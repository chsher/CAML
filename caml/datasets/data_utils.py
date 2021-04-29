import os
import sys
from os.path import dirname, realpath
sys.path.append(dirname(realpath(__file__)))
from caml.datasets import tcga

import numpy as np
import pandas as pd
from tqdm import tqdm

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
# ['BLCA', 'BRCA', 'COAD', 'HNSC', 'LUAD', 'LUSC', 'READ', 'STAD']
NORMALIZER = transforms.Normalize(mean=[0.7156, 0.5272, 0.6674], 
                                  std=[0.2002, 0.2330, 0.1982]) 

TRANSFORMER = transforms.Compose([transforms.ToPILImage(),
                                transforms.RandomVerticalFlip(),
                                transforms.RandomHorizontalFlip(),
                                transforms.ColorJitter(hue=0.02, saturation=0.1),
                                transforms.ToTensor(), NORMALIZER])

def build_transforms(mu, sig):
    normalizer = transforms.Normalize(mean=mu, std=sig) 

    transformer = transforms.Compose([transforms.ToPILImage(),
                                transforms.RandomVerticalFlip(),
                                transforms.RandomHorizontalFlip(),
                                transforms.ColorJitter(hue=0.02, saturation=0.1),
                                transforms.ToTensor(), normalizer])

    return transformer, transforms.Compose([normalizer])
    
#################### DATA SPLITTING ####################
def split_datasets_by_sample(df, train_frac=0.8, val_frac=0.2, n_pts=None, random_seed=31321, renormalize=False, min_tiles=1, 
                             num_tiles=100, cancers=None, label='WGD', unit='tile', mag='10.0', H=256, W=256, return_pt=False):
    '''
    Note: 
        - currently only handles TCGAdataset datasets

    Args:
        df (pandas.DataFrame): table with patient metadata (n_tiles, Type, n_tiles_start, n_tiles_end, basename)
        train_frac (float): fraction of examples allocated to the train set
        val_frac (float): fraction of examples allocated to the val set
        n_pts (int): number of patients to retain in the dataset
        random_seed (int): if not None, used to set the seed for numpy
        transform (torchvision.transforms.Compose): pytorch tensor transformations
        min_tiles (int): minimum number of tiles for patient to be included during sampling
        num_tiles (int): number of tiles to keep (tile) or sample (slide) per patient
        cancers (list or None): cancers to include in the dataset; if None, include all
        label (str): column name in df for label annotation
        unit (str): tile-level or slide-level inputs
        mag (str): magnification level of the images
        H (int): tile height
        W (int): tile width
    '''
    if random_seed is not None:
        np.random.seed(random_seed)
        
    df = filter_df(df, min_tiles, cancers, n_pts=n_pts, random_seed=random_seed)
        
    idxs = np.arange(df.shape[0])
    np.random.shuffle(idxs)
    
    if train_frac < 1:
        val_start = int(train_frac * len(idxs))
        dfs = [filter_df(df, idxs=idxs[:val_start])]
        
        if train_frac + val_frac < 1:
            test_start = int((train_frac + val_frac) * len(idxs))
            dfs.append(filter_df(df, idxs=idxs[val_start:test_start]))
            dfs.append(filter_df(df, idxs=idxs[test_start:]))
        else:
            dfs.append(filter_df(df, idxs=idxs[val_start:]))
    else:
        dfs = [filter_df(df, idxs=idxs)]

    if renormalize:
        ds = tcga.TCGAdataset(dfs[0], None, min_tiles, num_tiles, cancers, label, 'tile', mag, H, W, apply_filter=False, 
                              random_seed=random_seed, return_pt=return_pt)
        mu, sig = compute_stats(ds)
        transform_train, transform_val = build_transforms(mu, sig)
    else:
        transform_train = TRANSFORMER
        transform_val = transforms.Compose([NORMALIZER])
        
    dss = []
    for i, d in enumerate(dfs):
        if i == 0:
            ds = tcga.TCGAdataset(d, transform_train, min_tiles, num_tiles, cancers, label, unit, mag, H, W, apply_filter=False, 
                                  random_seed=random_seed, return_pt=return_pt) 
        else:
            ds = tcga.TCGAdataset(d, transform_val, min_tiles, num_tiles, cancers, label, unit, mag, H, W, apply_filter=False, 
                                  random_seed=random_seed, return_pt=return_pt) 
        dss.append(ds)
        
    return dss, transform_val.transforms[0].mean, transform_val.transforms[0].std

#################### DATA FILTERING ####################
def filter_df(df, min_tiles=None, cancers=None, idxs=None, n_pts=None, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
        
    if idxs is not None:
        df = df.loc[idxs, :]
        
    if min_tiles is not None:
        df = df[df['n_tiles'] >= min_tiles]
        
    if cancers is not None:
        df = df[df['Type'].isin(cancers)]
    
    if n_pts is not None and n_pts < df.shape[0]:
        pts = np.arange(df.shape[0])
        np.random.shuffle(pts)
        keep = pts[:n_pts]
        df = df.iloc[keep, :]
        
    df.reset_index(drop=True, inplace=True)
    
    df['n_tiles_end'] = df['n_tiles'].cumsum()
    df['n_tiles_start'] = np.concatenate((np.array([0]), df['n_tiles_end'].values[:-1]))
    
    return df

#################### STATS COMPUTING ####################
def compute_stats(train_dataset, batch_size=200, max_batches=240, pin_memory=False, num_workers=12):
    loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=True,
                        pin_memory=pin_memory, num_workers=num_workers)
    
    tracker = []
    with tqdm(total=min(len(loader), max_batches)) as pbar:
        for i, (x, y) in enumerate(loader):
            if i >= max_batches:
                break
            else:
                tracker.append(torch.mean(x, dim=[0, 2, 3]))
            pbar.update(1)
            
    means = torch.stack(tracker)
    mu = torch.mean(means, dim=0)
    
    tracker2 = []
    with tqdm(total=min(len(loader), max_batches)) as pbar:
        for i, (x, y) in enumerate(loader):
            if i >= max_batches:
                break
            else:
                t = torch.pow(x.transpose(1, -1) - mu, 2) # swapped dims 1 and 3 to enable broadcasting
                tracker2.append(torch.mean(t, dim=[0, 1, 2]))
            pbar.update(1)
            
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
    - convert image to float32 tensor
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