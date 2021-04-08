import os
import sys
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

#METADATA_FILEPATH = '/home/schao/url/results-20210308-203457_clean_031521.csv'

#TRAIN_CANCERS = ['BLCA', 'BRCA', 'COAD', 'HNSC', 'LUAD', 'LUSC', 'READ', 'STAD']
#VAL_CANCERS = ['ACC', 'CHOL', 'ESCA', 'LIHC', 'KICH', 'KIRC', 'OV', 'UCS', 'UCEC']

#################### SETUP ####################
args = script_utils.parse_args()

if not args.disable_cuda and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
#################### INIT DATA ####################
df = pd.read_csv(args.infile)

datasets, mu, sig = data_utils.split_datasets_by_sample(df, args.train_frac, args.val_frac, min_tiles=args.min_tiles, num_tiles=args.num_tiles,
                                                        unit=args.unit, cancers=args.cancers, renormalize=args.renormalize)
if args.test_val:
    train = datasets[0]
    transform_val = transforms.Compose([transforms.Normalize(mean=mu, std=sig)])
    val = tcga.TCGAdataset(df, transform=transform_val, min_tiles=args.min_tiles, num_tiles=args.num_tiles, unit=args.unit, cancers=args.val_cancers)
else:
    train, val = datasets

train_loader = DataLoader(train, batch_size=args.batch_size, pin_memory=args.pin_memory, num_workers=args.n_workers, shuffle=True, drop_last=True)
val_loader = DataLoader(val, batch_size=args.batch_size, pin_memory=args.pin_memory, num_workers=args.n_workers, shuffle=True, drop_last=False)

values = [args.renormalize, args.train_frac, args.val_frac, args.batch_size, args.wait_time, args.max_batches, args.pin_memory, args.n_workers, 
          args.training, args.learning_rate, args.weight_decay, args.dropout, args.patience, args.factor, args.n_epochs, args.disable_cuda, 
          args.output_size, args.min_tiles, args.num_tiles, args.unit, args.pool.__name__, ', '.join(args.cancers), args.infile, args.outfile, args.statsfile, 
          ', '.join(args.val_cancers), args.test_val, args.hidden_size, args.resfile, args.n_steps, args.n_testtrain, args.grad_adapt]
for k,v in zip(script_utils.PARAMS[:-2] + ['TRAIN_SIZE', 'VAL_SIZE', 'TRAIN_MU', 'TRAIN_SIG'], values + [len(train), len(val), mu, sig]):
    print('{0:12} {1}'.format(k, v))

#################### INIT MODEL ####################
net = feedforward.ClassifierNet(args.hidden_size, args.output_size, resfile=args.resfile, ffwdfile=args.outfile, dropout=args.dropout, freeze=True, pool=args.pool)
net.to(device)
print(net)

criterions = [nn.BCEWithLogitsLoss(reduction='mean'), nn.BCEWithLogitsLoss(reduction='none')]
optimizer = optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.factor, patience=args.patience, verbose=True)

#################### TRAIN ####################
learner.train_model(args.n_epochs, train_loader, [val_loader], net, criterions, optimizer, device, scheduler, args.patience, args.outfile, args.statsfile,
                    wait_time=args.wait_time, max_batches=args.max_batches, training=args.training, ff=True)

#with open(args.statsfile, 'ab') as f:
#    pickle.dump(stats, f)

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