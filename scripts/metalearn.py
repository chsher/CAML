import os
import sys
import datetime
from git import Repo
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
from tqdm import tqdm
from tqdm.contrib import tzip

#################### SETUP ####################
args = script_utils.parse_args()

if not args.disable_cuda and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
#################### INIT DATA ####################
df = pd.read_csv(args.infile)

assert args.n_testtest != 0
assert args.batch_size * args.wait_time == args.n_testtrain

dss = {'trains': [], 'vals': []}
for cas, lab in tzip([args.cancers, args.val_cancers], ['trains', 'vals']):
    for cancer in tqdm(cas):
        df_temp = data_utils.filter_df(df, min_tiles=args.min_tiles, cancers=[cancer])
        n_testtrain = df_temp.shape[0] - args.n_testtest

        tr_frac = n_testtrain / (n_testtrain + args.n_testtest)
        va_frac = 1.0 - tr_frac

        datasets, mu, sig = data_utils.split_datasets_by_sample(df, tr_frac, va_frac, random_seed=args.random_seed, renormalize=args.renormalize,
                                                                min_tiles=args.min_tiles, num_tiles=args.num_tiles, unit=args.unit, cancers=[cancer])
        dss[lab].append(datasets)

train_loaders = []
for tr in dss['trains']:
    tr_loader = DataLoader(tr[0], batch_size=args.batch_size, pin_memory=args.pin_memory, num_workers=args.n_workers, shuffle=True, drop_last=True)
    train_loaders.append(tr_loader)

metatrain_loaders = []
for va in dss['vals']:
    va_loader = DataLoader(va[0], batch_size=args.batch_size, pin_memory=args.pin_memory, num_workers=args.n_workers, shuffle=False, drop_last=True)
    metatrain_loaders.append(va_loader)

metatest_loaders = []
for va in dss['vals']:
    va_loader = DataLoader(va[1], batch_size=args.batch_size, pin_memory=args.pin_memory, num_workers=args.n_workers, shuffle=False, drop_last=True)
    metatest_loaders.append(va_loader)

val_loaders = [metatrain_loaders, metatest_loaders]
train_size = np.sum([len(tr[0]) for tr in dss['trains']])
metatrain_size = np.sum([len(va[0]) for va in dss['vals']])
metatest_size = np.sum([len(va[1]) for va in dss['vals']])

#################### PRINT PARAMS ####################
repo = Repo(search_parent_directories=True)
commit  = repo.head.object
commit_date = datetime.datetime.fromtimestamp(commit.committed_date)
print("Running CAML main as of commit:\n{}\ndesc: {}author: {}, date: {}".format(
    commit.hexsha, commit.message, commit.author, commit_date.strftime("%d-%b-%Y (%H:%M:%S)")))

values = [args.renormalize, args.train_frac, args.val_frac, args.batch_size, args.wait_time, args.max_batches, args.pin_memory, args.n_workers, args.random_seed, 
          args.training, args.learning_rate, args.weight_decay, args.dropout, args.patience, args.factor, args.n_epochs, args.disable_cuda, 
          args.output_size, args.min_tiles, args.num_tiles, args.unit, args.pool.__name__, ', '.join(args.cancers), args.infile, args.outfile, args.statsfile, 
          ', '.join(args.val_cancers), args.test_val, args.hidden_size, args.freeze, args.pretrained, args.resfile, args.resfile_new, args.grad_adapt, 
          args.eta, args.n_choose, args.n_steps, args.n_testtrain, args.n_testtest, args.randomize]
for k,v in zip(script_utils.PARAMS + ['TRAIN_SIZE', 'METATRAIN_SIZE', 'METATEST_SIZE'], values + [train_size, metatrain_size, metatest_size]):
    print('{0:12} {1}'.format(k, v))

#################### INIT MODEL ####################
net, global_model, local_models, global_theta = metalearner.init_models(args.hidden_size, args.output_size, len(args.cancers), device, dropout=args.dropout,
                                                                        resnet_file=args.resfile, maml_file=args.outfile, freeze=args.freeze)
print(net)
print(global_model)

criterions = [nn.BCEWithLogitsLoss(reduction='mean'), nn.BCEWithLogitsLoss(reduction='none')]

#################### TRAIN ####################
metalearner.train_model(args.n_epochs, train_loaders, val_loaders, args.learning_rate, args.eta, args.weight_decay, args.factor, 
                        net, global_model, local_models, global_theta, criterions, device, args.n_steps, args.n_testtrain, args.n_testtest, 
                        args.patience, args.outfile, args.statsfile, n_choose=args.n_choose, wait_time=args.wait_time, training=args.training, 
                        pool=args.pool, batch_size=args.batch_size, num_tiles=args.num_tiles, randomize=args.randomize)

#################### N_STEPS ####################
'''if args.test_val and args.grad_adapt:
    net, global_model, local_models, global_theta = metalearner.init_models(args.hidden_size, args.output_size, 0, device, dropout=args.dropout,
                                                                            resnet_file=args.resfile, maml_file=args.outfile, pool=args.pool)

    for s in [0, 1, 2, 3, 4]:
        for tt in [25, 50, 100, 150]:
            print('N_STEPS:', s, 'N_TESTTRAIN:', tt)
            stats = metalearner.train_model(1, train_loaders, val_loaders, args.learning_rate, args.eta, args.weight_decay, args.factor, net, global_model,
                                            local_models, global_theta, criterions, device, s, tt, args.patience, args.outfile, args.statsfile,
                                            n_choose=0, training=False)

            with open(args.statsfile, 'ab') as f:
                pickle.dump([s, tt, stats], f)'''