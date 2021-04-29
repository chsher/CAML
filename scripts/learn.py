import os
import sys
import datetime
from git import Repo
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))
from caml.datasets import tcga, data_utils
from caml.learn import learner
from caml.models import feedforward
from scripts import script_utils

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import models, transforms

import pickle
import numpy as np
import pandas as pd
import argparse

#################### SETUP ####################
args = script_utils.parse_args()

if not args.disable_cuda and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
#################### INIT DATA ####################
df = pd.read_csv(args.infile)

if args.n_testtest != 0:
    if args.n_testtrain == 0:
        df_temp = data_utils.filter_df(df, min_tiles=args.min_tiles, cancers=args.cancers)
        args.n_testtrain = df_temp.shape[0] - args.n_testtest
        
    tr_frac = args.n_testtrain / (args.n_testtrain + args.n_testtest)
    va_frac = 1.0 - tr_frac
    datasets, mu, sig = data_utils.split_datasets_by_sample(df, tr_frac, va_frac, random_seed=args.random_seed, renormalize=args.renormalize,
                                                            min_tiles=args.min_tiles, num_tiles=args.num_tiles, unit=args.unit, cancers=args.cancers,
                                                            n_pts=args.n_testtrain + args.n_testtest)
else:
    datasets, mu, sig = data_utils.split_datasets_by_sample(df, args.train_frac, args.val_frac, random_seed=args.random_seed, renormalize=args.renormalize,
                                                            min_tiles=args.min_tiles, num_tiles=args.num_tiles, unit=args.unit, cancers=args.cancers)
    
if args.test_val:
    train = datasets[0]
    transform_val = transforms.Compose([transforms.Normalize(mean=mu, std=sig)])
    val = tcga.TCGAdataset(df, transform=transform_val, min_tiles=args.min_tiles, num_tiles=args.num_tiles, unit=args.unit, 
                           cancers=args.val_cancers, random_seed=args.random_seed)
else:
    train, val = datasets

train_loader = DataLoader(train, batch_size=args.batch_size, pin_memory=args.pin_memory, num_workers=args.n_workers, shuffle=True, drop_last=True)
val_loader = DataLoader(val, batch_size=args.batch_size, pin_memory=args.pin_memory, num_workers=args.n_workers, shuffle=True, drop_last=False)

#################### PRINT PARAMS ####################
repo = Repo(search_parent_directories=True)
commit  = repo.head.object
commit_date = datetime.datetime.fromtimestamp(commit.committed_date)
print("Running CAML main as of commit:\n{}\ndesc: {}author: {}, date: {}".format(
    commit.hexsha, commit.message, commit.author, commit_date.strftime("%d-%b-%Y (%H:%M:%S)")))

values = [args.renormalize, args.train_frac, args.val_frac, args.batch_size, args.wait_time, args.max_batches, args.pin_memory, args.n_workers, args.random_seed,
          args.training, args.learning_rate, args.weight_decay, args.dropout, args.patience, args.factor, args.n_epochs, args.disable_cuda, 
          args.output_size, args.min_tiles, args.num_tiles, args.unit, args.pool.__name__, ', '.join(args.cancers), args.infile, args.outfile, args.statsfile, 
          ', '.join(args.val_cancers), args.test_val, args.hidden_size, args.freeze, args.pretrained, args.resfile, args.resfile_new, args.grad_adapt]
for k,v in zip(script_utils.PARAMS[:-6] + ['N_TRAIN', 'N_TEST', 'TRAIN_SIZE', 'VAL_SIZE', 'TRAIN_MU', 'TRAIN_SIG'], values + [args.n_testtrain, args.n_testtest, len(train), len(val), mu, sig]):
    print('{0:12} {1}'.format(k, v))

#################### INIT MODEL ####################
net = feedforward.ClassifierNet(args.hidden_size, args.output_size, resfile=args.resfile, ffwdfile=args.outfile, dropout=args.dropout, pool=args.pool, freeze=args.freeze, pretrained=args.pretrained)
net.to(device)
print(net)

criterions = [nn.BCEWithLogitsLoss(reduction='mean'), nn.BCEWithLogitsLoss(reduction='none')]
optimizer = optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.factor, patience=args.patience, verbose=True)

#################### TRAIN ####################
learner.train_model(args.n_epochs, train_loader, [val_loader], net, criterions, optimizer, device, scheduler, args.patience, args.outfile, args.statsfile,
                    resfile_new=args.resfile_new, wait_time=args.wait_time, max_batches=args.max_batches, training=args.training, ff=True, freeze=args.freeze)

'''#################### VAL - NO ADAPT ####################
if args.test_val:
    transform_val = transforms.Compose([transforms.Normalize(mean=mu, std=sig)])

    net = feedforward.ClassifierNet(args.hidden_size, args.output_size, resfile=args.resfile, ffwdfile=args.outfile, dropout=args.dropout, pool=args.pool)
    net.to(device)

    if not args.grad_adapt:
        va = tcga.TCGAdataset(df, transform=transform_val, min_tiles=args.min_tiles, num_tiles=args.num_tiles, unit=args.unit, cancers=args.val_cancers)
        va_loader = DataLoader(va, batch_size=args.batch_size, pin_memory=args.pin_memory, num_workers=args.n_workers, shuffle=True, drop_last=True)

        stats = learner.train_model(1, train_loader, [va_loader], net, criterions, optimizer, device, scheduler, args.patience, args.outfile,
                                    n_steps=0, n_testtrain=0, max_batches=args.max_batches, grad_adapt=args.grad_adapt, training=False)

        with open(args.statsfile, 'ab') as f:
            pickle.dump([0, 0, stats], f)

#################### VAL - ADAPT ####################
    elif args.grad_adapt:
        vals = []
        for cancer in args.val_cancers:
            va = tcga.TCGAdataset(df, transform=transform_val, min_tiles, args.min_tiles, num_tiles=args.num_tiles, unit=args.unit, cancers=[cancer])
            vals.append(va)

        val_loaders = []
        for va in vals:
            va_loader = DataLoader(va, batch_size=args.batch_size, pin_memory=args.pin_memory, num_workers=args.n_workers, shuffle=True, drop_last=True)
            val_loaders.append(va_loader)
                    
        for s in [0, 1, 2, 3, 4]:
            for tt in [25, 50, 100, 150]:
                print('N_STEPS:', s, 'N_TESTTRAIN:', tt)
                stats = learner.train_model(1, train_loader, val_loaders, net, criterions, optimizer, device, scheduler, args.patience, args.outfile,
                                            n_steps=s, n_testtrain=tt, max_batches=args.max_batches, grad_adapt=args.grad_adapt, training=False)

                with open(args.statsfile, 'ab') as f:
                    pickle.dump([s, tt, stats], f)'''