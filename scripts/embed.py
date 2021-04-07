import os
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))
from caml.datasets import data_utils
from caml.models import feedforward
from caml.learn import learner
from scripts import script_utils

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models

import pickle
import numpy as np
import pandas as pd
import argparse

#METADATA_FILEPATH = '/home/schao/url/results-20210308-203457_clean_031521.csv'

#CANCERS = ['BLCA', 'BRCA', 'COAD', 'HNSC', 'LUAD', 'LUSC', 'READ', 'STAD']

#################### SETUP ####################
args = script_utils.parse_args()

if not args.disable_cuda and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
#################### INIT DATA ####################
df = pd.read_csv(args.infile)

[train, val], mu, sig = data_utils.split_datasets_by_sample(df, args.train_frac, args.val_frac, min_tiles=args.min_tiles, num_tiles=args.num_tiles, unit=args.unit, cancers=args.cancers, renormalize=args.renormalize)
train_loader = DataLoader(train, batch_size=args.batch_size, pin_memory=args.pin_memory, num_workers=args.n_workers, shuffle=True, drop_last=True)
val_loader = DataLoader(val, batch_size=args.batch_size, pin_memory=args.pin_memory, num_workers=args.n_workers, shuffle=True, drop_last=False)

values = [args.renormalize, args.train_frac, args.val_frac, args.batch_size, args.wait_time, args.max_batches, args.pin_memory, args.n_workers, 
          args.training, args.learning_rate, args.weight_decay, args.dropout, args.patience, args.factor, args.n_epochs, args.disable_cuda, 
          args.output_size, args.min_tiles, args.num_tiles, args.unit, args.pool, ', '.join(args.cancers), args.infile, args.outfile, args.statsfile]
for k,v in zip(script_utils.PARAMS[:-9] + ['TRAIN_SIZE', 'VAL_SIZE', 'TRAIN_MU', 'TRAIN_SIG'], values + [len(train), len(val), mu, sig]):
    print('{0:12} {1}'.format(k, v))

#################### INIT MODEL ####################
#net = models.resnet18(pretrained=True)
#hidden_size = net.fc.weight.shape[1]
#net.fc = nn.Linear(hidden_size, args.output_size, bias=True)

#if os.path.exists(args.outfile):
#    saved_state = torch.load(args.outfile, map_location=lambda storage, loc: storage)
#    net.load_state_dict(saved_state)

net = feedforward.ClassifierNet(None, args.output_size, resfile=args.outfile, dropout=args.dropout, freeze=False, pool=args.pool)
net.to(device)
print(net)

criterions = [nn.BCEWithLogitsLoss(reduction='mean'), nn.BCEWithLogitsLoss(reduction='none')]
optimizer = optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.factor, patience=args.patience, verbose=True)

#################### TRAIN ####################
stats = learner.train_model(args.n_epochs, train_loader, [val_loader], net, criterions, optimizer, device, scheduler, args.patience, args.outfile, 
                            wait_time=args.wait_time, max_batches=args.max_batches, training=args.training)

with open(args.statsfile, 'wb') as f:
    pickle.dump(stats, f)             