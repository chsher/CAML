import os
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))
from caml.datasets import data_utils, maml
from caml.learn import metalearner

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

METADATA_FILEPATH = '/home/schao/url/results-20210308-203457_clean_031121.csv'

CANCERS = ['ESCA', 'LIHC', 'OV']

PARAMS = ['TRAIN_FRAC', 'VAL_FRAC', 'BATCH_SIZE', 'PIN_MEMORY', 'N_WORKERS', 'OUT_DIM', 'LR', 'WD', 
'PATIENCE', 'FACTOR', 'N_EPOCHS', 'DISABLE_CUDA', 'NUM_TILES', 'UNIT', 'CANCERS', 'METADATA', 'STATE_DICT', 'LOSS_STATS', 
'HID_DIM', 'RES_DICT', 'DROPOUT', 'ETA',
'TRAIN_SIZE', 'VAL_SIZE']

HIDDEN_SIZE = 512

#################### SETUP ####################
parser = argparse.ArgumentParser(description='Train H&E image meta-learner')

parser.add_argument('--train_frac', type=float, default=0.8, help='fraction of examples allocated to the train set')
parser.add_argument('--val_frac', type=float, default=0.2, help='fraction of examples allocated to the val set')
parser.add_argument('--batch_size', type=int, default=200, help='number of examples per batch')
parser.add_argument('--pin_memory', type=bool, default=True, help='whether to pin memory during data loading')
parser.add_argument('--n_workers', type=int, default=12, help='number of workers to use during data loading')
parser.add_argument('--output_size', type=int, default=1, help='model output dimension')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='local learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0, help='weight assigned to L2 regularization')
parser.add_argument('--patience', type=int, default=1, help='number of epochs with no improvement before invoking the scheduler, model reloading')
parser.add_argument('--factor', type=float, default=0.1, help='factor by which to reduce learning rate during scheduling')
parser.add_argument('--n_epochs', type=int, default=20, help='number of epochs to train the model')
parser.add_argument('--disable_cuda', type=bool, default=False, help='whether or not to use GPU')
parser.add_argument('--num_tiles', type=int, default=400, help='number of tiles to include per slide')
parser.add_argument('--unit', type=str, default='tile', help='input unit, i.e., whether to train on tile or slide')
parser.add_argument('--cancers', nargs='*', default=CANCERS, help='list of cancers to include')
parser.add_argument('--infile', type=str, default=METADATA_FILEPATH, help='file path to metadata dataframe')
parser.add_argument('--outfile', type=str, default='temp.pt', help='file path to save the model state dict')
parser.add_argument('--statsfile', type=float, default='temp.pkl', help='file path to save the per-epoch val stats')
# -- new params
parser.add_argument('--hidden_size', type=int, default=512, help='feed forward hidden size')
parser.add_argument('--resfile', type=str, default=None, help='path to pre-trained resnet')
parser.add_argument('--dropout', type=float, default=0.0, help='feed forward dropout')
parser.add_argument('--eta', type=float, default=0.01, help='global learning rate')
# --
args = parser.parse_args()

if not args.disable_cuda and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
#################### INIT DATA ####################
df = pd.read_csv(args.infile)

#test run
#train, val = data_utils.split_datasets_by_sample(df.loc[:2, :], 0.5, 0.5, unit='tile')

trains = []
vals = []
for cancer in args.cancers:
    tr, va = data_utils.split_datasets_by_sample(df, args.train_frac, args.val_frac, num_tiles=args.num_tiles, unit=args.unit, cancers=[cancer])
    trains.append(tr)
    vals.append(va)

train = maml.MAMLdataset(*trains)
val = maml.MAMLdataset(*vals)

train_loader = DataLoader(train, batch_size=args.batch_size, pin_memory=args.pin_memory, num_workers=args.n_workers, shuffle=True, drop_last=True)
val_loader = DataLoader(val, batch_size=args.batch_size, pin_memory=args.pin_memory, num_workers=args.n_workers, shuffle=True, drop_last=False)

values = [args.train_frac, args.val_frac, args.batch_size, args.pin_memory, args.n_workers, args.output_size, args.learning_rate, args.weight_decay, 
    args.patience, args.factor, args.n_epochs, args.disable_cuda, args.num_tiles, args.unit, ', '.join(args.cancers), args.infile, args.outfile, args.statsfile,
    args.hidden_size, args.resfile, args.dropout, args.eta]
for k,v in zip(PARAMS, values + [len(train), len(val)]):
    print('{0:12} {1}'.format(k, v))

#################### INIT MODEL ####################
net, global_model, local_models, global_theta = metalearner.init_models(HIDDEN_SIZE, args.hidden_size, args.output_size, len(args.cancers), args.device, dropout=args.dropout, resnet_file=args.resfile)
print(net)
print(global_model)

criterions = [nn.BCEWithLogitsLoss(reduction='mean'), nn.BCEWithLogitsLoss(reduction='none')]
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.patience, verbose=True)

#################### TRAIN ####################
stats = metalearner.train_model(args.n_epochs, train_loader, val_loader, args.learning_rate, args.eta, args.weight_decay, args.factor, net, global_model, local_models, theta_global, criterions, args.device, args.patience, args.outfile)

    learner.train_model(args.n_epochs, train_loader, val_loader, net, criterions, optimizer, device, scheduler, args.patience, args.outfile, ff=True)

with open(args.statsfile, 'wb') as f:
    pickle.dump(stats, f)
