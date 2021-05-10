import os
import sys
import datetime
from git import Repo
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

#################### SETUP ####################
args = script_utils.parse_args()

if not args.disable_cuda and torch.cuda.is_available():
    device = torch.device('cuda:' + args.device)
else:
    device = torch.device('cpu')
    
#################### INIT DATA ####################
df = pd.read_csv(args.infile)

[train, val], mu, sig = data_utils.split_datasets_by_sample(df, args.train_frac, args.val_frac, random_seed=args.random_seed, 
                                                            renormalize=args.renormalize, min_tiles=args.min_tiles, num_tiles=args.num_tiles, 
                                                            unit=args.unit, cancers=args.cancers, label=args.label)
train_loader = DataLoader(train, batch_size=args.batch_size, pin_memory=args.pin_memory, num_workers=args.n_workers, shuffle=True, drop_last=True)
val_loader = DataLoader(val, batch_size=args.batch_size, pin_memory=args.pin_memory, num_workers=args.n_workers, shuffle=True, drop_last=False)

#################### PRINT PARAMS ####################
repo = Repo(search_parent_directories=True)
commit  = repo.head.object
commit_date = datetime.datetime.fromtimestamp(commit.committed_date)
print("Running CAML main as of commit:\n{}\ndesc: {}author: {}, date: {}".format(
    commit.hexsha, commit.message, commit.author, commit_date.strftime("%d-%b-%Y (%H:%M:%S)")))

values = [args.renormalize, args.train_frac, args.val_frac, args.batch_size, args.wait_time, args.max_batches, args.pin_memory, args.n_workers, args.random_seed,
          args.training, args.learning_rate, args.weight_decay, args.dropout, args.patience, args.factor, args.n_epochs, args.disable_cuda, args.device,
          args.output_size, args.min_tiles, args.num_tiles, args.label, args.unit, args.pool.__name__, ', '.join(args.cancers), args.infile, args.outfile, args.statsfile]
for k,v in zip(script_utils.PARAMS[:-19] + ['RES_DICT', 'TRAIN_SIZE', 'VAL_SIZE', 'TRAIN_MU', 'TRAIN_SIG'], values + [args.resfile, len(train), len(val), mu, sig]):
    print('{0:12} {1}'.format(k, v))

#################### INIT MODEL ####################
net = feedforward.ClassifierNet(None, args.output_size, resfile=args.resfile, dropout=args.dropout, pool=args.pool, freeze=False, pretrained=True)
net.to(device)
print(net)

criterions = [nn.BCEWithLogitsLoss(reduction='mean'), nn.BCEWithLogitsLoss(reduction='none')]
optimizer = optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.factor, patience=args.patience, eps=1e-12, verbose=True)

#################### TRAIN ####################
learner.train_model(args.n_epochs, train_loader, [val_loader], net, criterions, optimizer, device, scheduler, args.patience, args.outfile, args.statsfile, 
                    wait_time=args.wait_time, max_batches=args.max_batches, training=args.training, freeze=False)          