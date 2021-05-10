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
    device = torch.device('cuda:' + args.device)
else:
    device = torch.device('cpu')

np.random.seed(args.random_seed)

#################### INIT DATA ####################
df = pd.read_csv(args.infile)

df_temp = data_utils.filter_df(df, min_tiles=args.min_tiles, cancers=args.cancers)

if args.test_val:
    tr_frac = 1.0
    va_frac = 0.0
    n_pts = df_temp.shape[0]
else:
    tr_frac, va_frac, n_pts = data_utils.compute_fracs(df_temp, args.n_testtrain, args.n_testtest, args.train_frac, args.val_frac)

datasets, mu, sig = data_utils.split_datasets_by_sample(df, train_frac=tr_frac, val_frac=va_frac, n_pts=n_pts, random_seed=args.random_seed, 
                                                        renormalize=args.renormalize, min_tiles=args.min_tiles, num_tiles=args.num_tiles, 
                                                        unit=args.unit, cancers=args.cancers, label=args.label)
    
if args.test_val:
    train = datasets[0]
    
    metatrain_loaders = []
    metatest_loaders = []

    for cancer in args.val_cancers:
        df_temp = data_utils.filter_df(df, min_tiles=args.min_tiles, cancers=[cancer])
        tr_frac, va_frac, n_pts = data_utils.compute_fracs(df_temp, args.n_testtrain, args.n_testtest, args.train_frac, args.val_frac)

        if args.training:
            datasets = []
            rands = np.random.randint(0, 1e09, size=args.n_replicates)

            for randseed in rands:
                datasets_v, mu, sig = data_utils.split_datasets_by_sample(df, train_frac=tr_frac, val_frac=va_frac, n_pts=n_pts, random_seed=randseed, 
                                                                renormalize=args.renormalize, min_tiles=args.min_tiles, num_tiles=args.num_tiles, 
                                                                unit=args.unit, cancers=[cancer], label=args.label,
                                                                adjust_brightness=args.adjust_brightness, resize=args.resize)
                datasets.append(datasets_v)
            
            metatrain_loaders.append([DataLoader(v[0], batch_size=args.batch_size, pin_memory=args.pin_memory, num_workers=args.n_workers, 
                                                 shuffle=True, drop_last=True) for v in datasets])
            metatest_loaders.append([DataLoader(v[1], batch_size=args.test_batch_size, pin_memory=args.pin_memory, num_workers=args.n_workers, 
                                                shuffle=True, drop_last=False) for v in datasets])
        else:
            datasets, mu, sig = data_utils.split_datasets_by_sample(df, train_frac=tr_frac, val_frac=va_frac, n_pts=n_pts, random_seed=args.random_seed, 
                                                                renormalize=args.renormalize, min_tiles=args.min_tiles, num_tiles=args.num_tiles, 
                                                                unit=args.unit, cancers=[cancer], label=args.label,
                                                                adjust_brightness=args.adjust_brightness, resize=args.resize)
            
            metatrain_loaders.append([DataLoader(datasets[0], batch_size=args.batch_size, pin_memory=args.pin_memory, num_workers=args.n_workers, 
                                                 shuffle=True, drop_last=True)])
            metatest_loaders.append([DataLoader(datasets[1], batch_size=args.test_batch_size, pin_memory=args.pin_memory, num_workers=args.n_workers, 
                                                shuffle=True, drop_last=False)])

    val_loaders = [metatrain_loaders, metatest_loaders]
else:
    train, val = datasets
    val_loader = DataLoader(val, batch_size=args.test_batch_size, pin_memory=args.pin_memory, num_workers=args.n_workers, shuffle=True, drop_last=False)
    val_loaders = [val_loader]

train_loader = DataLoader(train, batch_size=args.batch_size, pin_memory=args.pin_memory, num_workers=args.n_workers, shuffle=True, drop_last=True)    
args.max_batches = [args.max_batches[0], args.max_batches[0]] if len(args.max_batches) == 1 else args.max_batches

#################### PRINT PARAMS ####################
repo = Repo(search_parent_directories=True)
commit  = repo.head.object
commit_date = datetime.datetime.fromtimestamp(commit.committed_date)
print("Running CAML main as of commit:\n{}\ndesc: {}author: {}, date: {}".format(
    commit.hexsha, commit.message, commit.author, commit_date.strftime("%d-%b-%Y (%H:%M:%S)")))

values = [args.renormalize, args.train_frac, args.val_frac, args.batch_size, args.wait_time, args.max_batches, args.pin_memory, args.n_workers, args.random_seed,
          args.training, args.learning_rate, args.weight_decay, args.dropout, args.patience, args.factor, args.n_epochs, args.disable_cuda, args.device,
          args.output_size, args.min_tiles, args.num_tiles, args.label, args.unit, args.pool.__name__, ', '.join(args.cancers), args.infile, args.outfile, args.statsfile, 
          ', '.join(args.val_cancers), args.test_val, args.hidden_size, args.freeze, args.pretrained, args.resfile, args.resfile_new, args.grad_adapt]
for k,v in zip(script_utils.PARAMS[:-11] + ['N_TRAIN', 'N_TEST', 'N_REPLICATES', 'TEST_BATCH_SIZE', 'TRAIN_SIZE', 'TRAIN_MU', 'TRAIN_SIG'], values + [args.n_testtrain, args.n_testtest, args.n_replicates, args.test_batch_size, len(train), mu, sig]):
    print('{0:12} {1}'.format(k, v))

#################### INIT MODEL ####################
ff = 'ff' in args.outfile
if ff:
    net = feedforward.ClassifierNet(args.hidden_size, args.output_size, resfile=args.resfile, ffwdfile=args.outfile, dropout=args.dropout, pool=args.pool, freeze=args.freeze, pretrained=args.pretrained)
else:
    net = feedforward.ClassifierNet(args.hidden_size, args.output_size, resfile=args.outfile, ffwdfile=None, dropout=args.dropout, pool=args.pool, freeze=args.freeze, pretrained=args.pretrained)
    
net.to(device)
print(net)

criterions = [nn.BCEWithLogitsLoss(reduction='mean'), nn.BCEWithLogitsLoss(reduction='none')]
optimizer = optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.factor, patience=args.patience, eps=1e-12, verbose=True)

#################### TRAIN ####################
learner.train_model(args.n_epochs, train_loader, val_loaders, net, criterions, optimizer, device, scheduler,args.patience, args.outfile, args.statsfile, 
                    resfile_new=args.resfile_new, n_steps=args.n_steps, wait_time=args.wait_time, pool=args.pool, batch_size=args.batch_size, 
                    num_tiles=args.num_tiles,max_batches=args.max_batches, grad_adapt=args.grad_adapt, training=args.training, ff=ff, freeze=args.freeze)

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