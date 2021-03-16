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
import argparse

METADATA_FILEPATH = '/home/schao/url/results-20210308-203457_clean_031521.csv'

CANCERS = ['BLCA', 'BRCA', 'COAD', 'HNSC', 'LUAD', 'LUSC', 'READ', 'STAD']

PARAMS = ['TRAIN_FRAC', 'VAL_FRAC', 'BATCH_SIZE', 'WAIT_TIME', 'MAX_BATCHES', 'PIN_MEMORY', 'N_WORKERS', 
          'OUT_DIM', 'LR', 'WD', 'PATIENCE', 'FACTOR', 'N_EPOCHS', 'DISABLE_CUDA', 
          'NUM_TILES', 'UNIT', 'CANCERS', 'METADATA', 'STATE_DICT', 'LOSS_STATS', 'TRAINING'
          'TRAIN_SIZE', 'VAL_SIZE']

#################### SETUP ####################
parser = argparse.ArgumentParser(description='Embed H&E images using ResNet18 pre-trained on ImageNet')

parser.add_argument('--train_frac', type=float, default=0.8, help='fraction of examples allocated to the train set')
parser.add_argument('--val_frac', type=float, default=0.2, help='fraction of examples allocated to the val set')
parser.add_argument('--batch_size', type=int, default=200, help='number of examples per batch')
parser.add_argument('--wait_time', type=int, default=2, help='number of batches before backward pass')
parser.add_argument('--max_batches', type=int, default=-1, help='max number of batches per epoch (-1: include all)')
parser.add_argument('--pin_memory', type=bool, default=True, help='whether to pin memory during data loading')
parser.add_argument('--n_workers', type=int, default=12, help='number of workers to use during data loading')

parser.add_argument('--output_size', type=int, default=1, help='model output dimension')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate (step size)')
parser.add_argument('--weight_decay', type=float, default=50.0, help='weight assigned to L2 regularization')
parser.add_argument('--patience', type=int, default=1, help='number of epochs with no improvement before invoking the scheduler, model reloading')
parser.add_argument('--factor', type=float, default=0.1, help='factor by which to reduce learning rate during scheduling')
parser.add_argument('--n_epochs', type=int, default=10, help='number of epochs to train the model')
parser.add_argument('--disable_cuda', type=bool, default=False, help='whether or not to use GPU')

parser.add_argument('--num_tiles', type=int, default=100, help='number of tiles to include per slide')
parser.add_argument('--unit', type=str, default='tile', help='input unit, i.e., whether to train on tile or slide')
parser.add_argument('--cancers', nargs='*', default=CANCERS, help='list of cancers to include')
parser.add_argument('--infile', type=str, default=METADATA_FILEPATH, help='file path to metadata dataframe')
parser.add_argument('--outfile', type=str, default='temp.pt', help='file path to save the model state dict')
parser.add_argument('--statsfile', type=str, default='temp.pkl', help='file path to save the per-epoch val stats')
parser.add_argument('--training', type=bool, default=True, help='whether to train the model')

args = parser.parse_args()

if not args.disable_cuda and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
#################### INIT DATA ####################
df = pd.read_csv(args.infile)

train, val = data_utils.split_datasets_by_sample(df, args.train_frac, args.val_frac, num_tiles=args.num_tiles, unit=args.unit, cancers=args.cancers)
train_loader = DataLoader(train, batch_size=args.batch_size, pin_memory=args.pin_memory, num_workers=args.n_workers, shuffle=True, drop_last=True)
val_loader = DataLoader(val, batch_size=args.batch_size, pin_memory=args.pin_memory, num_workers=args.n_workers, shuffle=True, drop_last=False)

values = [args.train_frac, args.val_frac, args.batch_size, args.wait_time, args.max_batches, args.pin_memory, args.n_workers, 
          args.output_size, args.learning_rate, args.weight_decay, args.patience, args.factor, args.n_epochs, args.disable_cuda, 
          args.num_tiles, args.unit, ', '.join(args.cancers), args.infile, args.outfile, args.statsfile, args.training]
for k,v in zip(PARAMS, values + [len(train), len(val)]):
    print('{0:12} {1}'.format(k, v))

#################### INIT MODEL ####################
net = models.resnet18(pretrained=True)
hidden_size = net.fc.weight.shape[1]
net.fc = nn.Linear(hidden_size, args.output_size, bias=True)

if os.path.exists(args.outfile):
    saved_state = torch.load(args.outfile, map_location=lambda storage, loc: storage)
    net.load_state_dict(saved_state)

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