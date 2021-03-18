import os
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))
from caml.datasets import tcga, data_utils
from caml.learn import metalearner
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

METADATA_FILEPATH = '/home/schao/url/results-20210308-203457_clean_031521.csv'

TRAIN_CANCERS = ['BLCA', 'BRCA', 'COAD', 'HNSC', 'LUAD', 'LUSC', 'READ', 'STAD']
VAL_CANCERS = ['ACC', 'CHOL', 'ESCA', 'LIHC', 'KICH', 'KIRC', 'OV', 'UCS', 'UCEC']

PARAMS = ['TRAIN_FRAC', 'VAL_FRAC', 'BATCH_SIZE', 'WAIT_TIME', 'MAX_BATCHES', 'PIN_MEMORY', 'N_WORKERS', 
          'OUT_DIM', 'LR', 'WD', 'PATIENCE', 'FACTOR', 'N_EPOCHS', 'DISABLE_CUDA', 
          'NUM_TILES', 'UNIT', 'CANCERS', 'METADATA', 'STATE_DICT', 'LOSS_STATS', 'TRAINING',
          'VAL_CANCERS', 'HID_DIM', 'RES_DICT', 'DROPOUT', 'N_STEPS', 'N_TESTTRAIN', 'ETA', 'N_CHOOSE',
          'TRAIN_SIZE', 'VAL_SIZE']

#################### SETUP ####################
args = script_utils.parse_args()

if not args.disable_cuda and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
#################### INIT DATA ####################
df = pd.read_csv(args.infile)

trains = []
for cancer in args.cancers:
    tr = tcga.TCGAdataset(df, transform=data_utils.transform, num_tiles=args.num_tiles, unit=args.unit, cancers=[cancer])
    trains.append(tr)

vals = []
for cancer in args.val_cancers:
    va = tcga.TCGAdataset(df, transform=transforms.Compose([data_utils.normalize]), num_tiles=args.num_tiles, unit=args.unit, cancers=[cancer])
    vals.append(va)

train_loaders = []
for tr in trains:
    tr_loader = DataLoader(tr, batch_size=args.batch_size, pin_memory=args.pin_memory, num_workers=args.n_workers, shuffle=True, drop_last=True)
    train_loaders.append(tr_loader)

val_loaders = []
for va in vals:
    va_loader = DataLoader(va, batch_size=args.batch_size, pin_memory=args.pin_memory, num_workers=args.n_workers, shuffle=True, drop_last=True)
    val_loaders.append(va_loader)

values = [args.train_frac, args.val_frac, args.batch_size, args.wait_time, args.max_batches, args.pin_memory, args.n_workers, 
          args.output_size, args.learning_rate, args.weight_decay, args.patience, args.factor, args.n_epochs, args.disable_cuda, 
          args.num_tiles, args.unit, ', '.join(args.cancers), args.infile, args.outfile, args.statsfile, args.training,
          ', '.join(args.val_cancers), args.hidden_size, args.resfile, args.dropout, args.n_steps, args.n_testtrain, args.eta, args.n_choose]
for k,v in zip(PARAMS, values + [np.sum([len(tr) for tr in trains]), np.sum([len(va) for va in vals])]):
    print('{0:12} {1}'.format(k, v))

#################### INIT MODEL ####################
net, global_model, local_models, global_theta = metalearner.init_models(args.hidden_size, args.output_size, len(args.cancers), device, dropout=args.dropout,
                                                                        resnet_file=args.resfile, maml_file=args.outfile)
print(net)
print(global_model)

criterions = [nn.BCEWithLogitsLoss(reduction='mean'), nn.BCEWithLogitsLoss(reduction='none')]

#################### TRAIN ####################
if args.training:
    stats = metalearner.train_model(args.n_epochs, train_loaders, val_loaders, args.learning_rate, args.eta, args.weight_decay, args.factor, net, global_model,
                                local_models, global_theta, criterions, device, args.n_steps, args.n_testtrain, args.patience, args.outfile, n_choose=args.n_choose,
                                training=args.training)

    with open(args.statsfile, 'ab') as f:
        pickle.dump(stats, f)
    
#################### N_STEPS ####################
for s in [0, 1, 2, 3, 4]:
    for tt in [25, 50, 100, 150]:
        print('N_STEPS:', s, 'N_TESTTRAIN:', tt)
        stats = metalearner.train_model(1, train_loaders, val_loaders, args.learning_rate, args.eta, args.weight_decay, args.factor, net, global_model,
                                        local_models, global_theta, criterions, device, s, tt, args.patience, args.outfile, n_choose=args.n_choose,
                                        training=False)

        with open(args.statsfile, 'ab') as f:
            pickle.dump([s, tt, stats], f)