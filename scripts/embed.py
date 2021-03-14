import os
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))
from caml.datasets import data_utils
from caml.learn import learner

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models

import pickle
import numpy as np
import pandas as pd

METADATA_FILEPATH = '/home/schao/url/results-20210308-203457_clean_031121.csv'
params = ['TRAIN_FRAC', 'VAL_FRAC', 'BATCH_SIZE', 'PIN_MEMORY', 'N_WORKERS', 'OUT_DIM', 'LR', 'WD', 'PATIENCE', 'N_EPOCHS', 'DISABLE_CUDA', 'NUM_TILES', 'STATE_DICT', 'LOSS_STATS', 'TRAIN_SIZE', 'VAL_SIZE']

#################### SETUP ####################
train_frac = 0.8
val_frac = 0.2
batch_size = 200
pin_memory = True
n_workers = 12
output_size = 1
learning_rate = 0.0001
weight_decay = 50.0
patience = 1
n_epochs = 20
disable_cuda = False
num_tiles = 400
outfile = '/home/schao/dev/output/embed_state_dict_031221_v02.pt'
statsfile = '/home/schao/dev/output/embed_loss_stats_031221_v02.pkl'

if not disable_cuda and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
#################### INIT DATA ####################
df = pd.read_csv(METADATA_FILEPATH)

#test run
#train, val = data_utils.split_datasets_by_sample(df.loc[:2, :], 0.5, 0.5, unit='tile')

train, val = data_utils.split_datasets_by_sample(df, train_frac, val_frac, num_tiles=num_tiles, unit='tile', cancers=['BLCA', 'BRCA', 'COAD', 'HNSC', 'LUAD', 'LUSC', 'READ', 'STAD'])
train_loader = DataLoader(train, batch_size=batch_size, pin_memory=pin_memory, num_workers=n_workers, shuffle=True, drop_last=True)
val_loader = DataLoader(val, batch_size=batch_size, pin_memory=pin_memory, num_workers=n_workers, shuffle=True, drop_last=False)

for k,v in zip(params, [train_frac, val_frac, batch_size, pin_memory, n_workers, output_size, learning_rate, weight_decay, patience, n_epochs, disable_cuda, num_tiles, outfile, statsfile, len(train), len(val)]):
    print('{0:12} {1}'.format(k, v))

#################### INIT MODEL ####################
net = models.resnet18(pretrained=True)
hidden_size = net.fc.weight.shape[1]
net.fc = nn.Linear(hidden_size, output_size, bias=True)
print(net)
net.to(device)
#torch.save(net.state_dict(), outfile)

criterions = [nn.BCEWithLogitsLoss(reduction='mean'), nn.BCEWithLogitsLoss(reduction='none')]
optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, verbose=True)

#################### TRAIN ####################
stats = learner.train_model(n_epochs, train_loader, val_loader, net, criterions, optimizer, device, scheduler, patience, outfile)

with open(statsfile, 'wb') as f:
    pickle.dump(stats, f)
