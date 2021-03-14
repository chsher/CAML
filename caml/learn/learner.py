import os
import sys
from os.path import dirname, realpath
sys.path.append(dirname(realpath(__file__)))
from caml.learn import metalearner

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

PRINT_STMT = 'Epoch {0:3d}, Minibatch {1:3d}, {6:6} Loss {2:7.4f} AUC {3:7.4f}, {7:6} Loss {4:7.4f} AUC {5:7.4f}'

def train_model(n_epochs, train_loader, val_loaders, net, criterions, optimizer, device, n_steps, scheduler, patience, outfile, verbose=True, ff=True):
    tally = 0
    old_loss = 1e9
    overall_loss_tracker = []
    overall_auc_tracker = []
    
    alpha = optimizer.param_groups[0]['lr']
    wd = optimizer.param_groups[0]['weight_decay']

    for n in tqdm(range(n_epochs)):
        run_training_epoch(n, train_loader, val_loader, net, criterions[0], optimizer, device, verbose)
        
        global_theta = []
        for p in net.ff.parameters():
            global_theta.append(p.detach().clone().to(device))

        loss, auc = metalearner.run_validation(n, val_loaders, alpha, wd, net.resnet, net.ff, global_theta, criterions, device, n_steps, verbose)
        overall_loss_tracker.append(loss)
        overall_auc_tracker.append(auc)
        
        if loss < old_loss: 
            old_loss = loss 
            if ff:
                torch.save(net.ff.state_dict(), outfile)
            else:
                torch.save(net.state_dict(), outfile)
            print('----- SAVED MODEL -----')
        else:
            tally += 1
                
        if scheduler is not None:
            scheduler.step(loss)

        if tally > patience:
            saved_state = torch.load(outfile, map_location=lambda storage, loc: storage)
            if ff:
                net.ff.load_state_dict(saved_state)
            else:
                net.load_state_dict(saved_state)
            print('----- RELOADED MODEL -----')
            tally = 0
            
    return overall_loss_tracker, overall_auc_tracker

def cycle(iterable):
    '''
    - source: https://github.com/pytorch/pytorch/issues/23900
    - addresses memory leak issue from itertools cycle
    '''
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)
            
def run_training_epoch(epoch_num, train_loader, val_loader, net, criterion, optimizer, device, verbose=True, splits=['Train', 'Val'], wait_time=2):
    total_loss = 0.0
    
    for t, ((x, y), (x_val, y_val)) in enumerate(zip(tqdm(train_loader), cycle(val_loader))):
        net.train()
        optimizer.zero_grad()
        
        y_pred = net(x.to(device))
        loss = criterion(y_pred, y.to(device))
        total_loss += loss

        if (t + 1) % wait_time == 0:
            total_loss.backward()
            optimizer.step()
            total_loss = 0.0
            
            y_prob = torch.sigmoid(y_pred.detach().cpu())
            try:
                auc = roc_auc_score(y.squeeze(-1).numpy(), y_prob.squeeze(-1).numpy())
            except:
                auc = 0.0
            
            net.eval()
            with torch.no_grad():
                y_pred_val = net(x_val.to(device))
                loss_val = criterion(y_pred_val, y_val.to(device))

                y_prob_val = torch.sigmoid(y_pred_val.detach().cpu())
                try:
                    auc_val = roc_auc_score(y_val.squeeze(-1).numpy(), y_prob_val.squeeze(-1).numpy())
                except:
                    auc_val = 0.0

            if verbose:
                print(PRINT_STMT.format(epoch_num, t, loss.item(), auc, loss_val, auc_val, *splits))