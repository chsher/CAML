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
import pandas as pd

METADATA_FILEPATH = '/home/schao/url/results-20210308-203457_clean_031121.csv'
params = ['OUT_DIM', 'LR', 'WD', 'PATIENCE', 'N_EPOCHS', 'STATE_DICT', 'LOSS_STATS', 'DISABLE_CUDA']

#################### SETUP ####################
output_size = 1
learning_rate = 0.001
weight_decay = 0.001
patience = 2
n_epochs = 10
outfile = '/home/schao/dev/output/embed_state_dict_031221_v01.pt'
statsfile = '/home/schao/dev/output/embed_loss_stats_031221_v01.pkl'

disable_cuda = False
if not disable_cuda and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

for k,v in zip(params, [output_size, learning_rate, weight_decay, patience, n_epochs, outfile, statsfile, disable_cuda]):
    print('{0:12} {1}'.format(k, v))
    
#################### INIT DATA ####################
df = pd.read_csv(METADATA_FILEPATH)
train, val = data_utils.split_datasets_by_sample(df, 0.8, 0.2, unit='tile', cancers=['BLCA', 'BRCA', 'COAD', 'HNSC', 'LUAD', 'LUSC', 'READ', 'STAD'])
train_loader = DataLoader(train, batch_size=250, drop_last=True, pin_memory=True, num_workers=3)
val_loader = DataLoader(val, batch_size=250, drop_last=False, pin_memory=True, num_workers=3)

# test run
#train, val = data_utils.split_datasets_by_sample(df.loc[:2, :], 0.5, 0.5, unit='tile')#, cancers=['BLCA', 'BRCA', 'COAD', 'HNSC', 'LUAD', 'LUSC', 'READ', 'STAD'])
#train_loader = DataLoader(train, batch_size=250, drop_last=True, pin_memory=True, num_workers=4)
#val_loader = DataLoader(val, batch_size=250, drop_last=False, pin_memory=True, num_workers=4)

#################### INIT MODEL ####################
net = models.resnet18(pretrained=True)
hidden_size = net.fc.weight.shape[1]
net.fc = nn.Linear(hidden_size, output_size, bias=True)
print(net)
net.to(device)
torch.save(net.state_dict(), outfile)

criterions = [nn.BCEWithLogitsLoss(reduction='mean'), nn.BCEWithLogitsLoss(reduction='none')]
optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, verbose=True)

#################### TRAIN ####################
stats = learner.train_model(n_epochs, train_loader, val_loader, net, criterions, optimizer, device, scheduler, patience, outfile)

with open(statsfile, 'wb') as f:
    pickle.dump(stats, f)