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
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

import pickle
import numpy as np
import pandas as pd
import argparse

METADATA_FILEPATH = '/home/schao/url/results-20210308-203457_clean_031521.csv'

TRAIN_CANCERS = ['BLCA', 'BRCA', 'COAD', 'HNSC', 'LUAD', 'LUSC', 'READ', 'STAD']
VAL_CANCERS = ['ACC', 'CHOL', 'ESCA', 'LIHC', 'KICH', 'KIRC', 'OV', 'UCS', 'UCEC']

PARAMS = ['TRAIN_FRAC', 'VAL_FRAC', 'BATCH_SIZE', 'WAIT_TIME', 'MAX_BATCHES', 'PIN_MEMORY', 'N_WORKERS', 
          'OUT_DIM', 'LR', 'WD', 'PATIENCE', 'FACTOR', 'N_EPOCHS', 'DISABLE_CUDA', 
          'NUM_TILES', 'UNIT', 'CANCERS', 'METADATA', 'STATE_DICT', 'LOSS_STATS', 'TRAINING',
          'VAL_CANCERS', 'HID_DIM', 'RES_DICT', 'DROPOUT', 'N_STEPS', 'N_TESTTRAIN',
          'TRAIN_SIZE', 'VAL_SIZE']

#################### SETUP ####################
args = script_utils.parse_args()

if not args.disable_cuda and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
#################### INIT DATA ####################
df = pd.read_csv(args.infile)

train, val = data_utils.split_datasets_by_sample(df, args.train_frac, args.val_frac, num_tiles=args.num_tiles, unit=args.unit, cancers=args.cancers)
train_loader = DataLoader(train, batch_size=args.batch_size, pin_memory=args.pin_memory, num_workers=args.n_workers, shuffle=True, drop_last=True)
val_loader = DataLoader(val, batch_size=args.batch_size, pin_memory=args.pin_memory, num_workers=args.n_workers, shuffle=True, drop_last=True)

values = [args.train_frac, args.val_frac, args.batch_size, args.wait_time, args.max_batches, args.pin_memory, args.n_workers, 
          args.output_size, args.learning_rate, args.weight_decay, args.patience, args.factor, args.n_epochs, args.disable_cuda, 
          args.num_tiles, args.unit, ', '.join(args.cancers), args.infile, args.outfile, args.statsfile, args.training,
          ', '.join(args.val_cancers), args.hidden_size, args.resfile, args.dropout, args.n_steps, args.n_testtrain]
for k,v in zip(PARAMS, values + [len(train), len(val)]):
    print('{0:12} {1}'.format(k, v))

#################### INIT MODEL ####################
net = feedforward.ClassifierNet(args.hidden_size, args.output_size, resfile=args.resfile, ffwdfile=args.outfile, dropout=args.dropout)
net.to(device)
print(net)

criterions = [nn.BCEWithLogitsLoss(reduction='mean'), nn.BCEWithLogitsLoss(reduction='none')]
optimizer = optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.factor, patience=args.patience, verbose=True)

#################### TRAIN ####################
if args.training:
    stats = learner.train_model(args.n_epochs, train_loader, [val_loader], net, criterions, optimizer, device, scheduler, args.patience, args.outfile,
                                n_steps=args.n_steps, n_testtrain=args.n_testtrain, wait_time=args.wait_time, max_batches=args.max_batches, grad_adapt=False, 
                                training=args.training, ff=True)

    with open(args.statsfile, 'ab') as f:
        pickle.dump(stats, f)

#################### VAL - NO ADAPT ####################
net = feedforward.ClassifierNet(args.hidden_size, args.output_size, resfile=args.resfile, ffwdfile=args.outfile, dropout=args.dropout)
net.to(device)

va = tcga.TCGAdataset(df, transform=transforms.Compose([data_utils.normalize]), num_tiles=args.num_tiles, unit=args.unit, cancers=args.val_cancers)
va_loader = DataLoader(va, batch_size=args.batch_size, pin_memory=args.pin_memory, num_workers=args.n_workers, shuffle=True, drop_last=True)

stats = learner.run_validation_epoch(0, va_loader, net, criterions[1], device, max_batches=args.max_batches * 10)
with open(args.statsfile, 'ab') as f:
    pickle.dump([0, 0, stats], f)

#################### VAL - ADAPT ####################
vals = []
for cancer in args.val_cancers:
    va = tcga.TCGAdataset(df, transform=transforms.Compose([data_utils.normalize]), num_tiles=args.num_tiles, unit=args.unit, cancers=[cancer])
    vals.append(va)

val_loaders = []
for va in vals:
    va_loader = DataLoader(va, batch_size=args.batch_size, pin_memory=args.pin_memory, num_workers=args.n_workers, shuffle=True, drop_last=True)
    val_loaders.append(va_loader)
            
for s in [0, 1, 2, 3, 4]:
    for tt in [25, 50, 100, 150]:
        print('N_STEPS:', s, 'N_TESTTRAIN:', tt)
        stats = learner.train_model(1, train_loader, val_loaders, net, criterions, optimizer, device, scheduler, args.patience, args.outfile,
                                    n_steps=s, n_testtrain=tt, wait_time=args.wait_time, max_batches=args.max_batches, grad_adapt=True, training=False)

        with open(args.statsfile, 'ab') as f:
            pickle.dump([s, tt, stats], f)