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

def train_model(n_epochs, train_loader, val_loaders, net, criterions, optimizer, device, scheduler, patience, outfile, n_steps=1, n_testtrain=50, 
                wait_time=1, max_batches=20, grad_adapt=False, ff=False, training=True, verbose=True):
    tally = 0
    best_n = 0
    old_loss = 1e9
    overall_loss_tracker = []
    overall_auc_tracker = []
    y_tracker = []
    y_prob_tracker = []
    
    alpha = optimizer.param_groups[0]['lr']
    wd = optimizer.param_groups[0]['weight_decay']

    for n in tqdm(range(n_epochs)):
        if training:
            run_training_epoch(n, train_loader, val_loaders[0], net, criterions[0], optimizer, device, verbose, wait_time, max_batches)
        
        if grad_adapt:
            global_theta = []
            for p in net.ff.parameters():
                global_theta.append(p.detach().clone().to(device))
            loss, auc, ys, yps = metalearner.run_validation(n, val_loaders, alpha, wd, net.resnet, net.ff, global_theta, criterions, device, n_steps, 
                                                            n_testtrain, verbose)
        else:
            loss, auc, ys, yps = run_validation_epoch(n, val_loaders[0], net, criterions[1], device, verbose, max_batches)

        overall_loss_tracker.append(loss)
        overall_auc_tracker.append(auc)
        y_tracker.append(ys)
        y_prob_tracker.append(yps)
        
        if loss < old_loss: 
            best_n = n
            old_loss = loss 

        if training:
            if loss < old_loss: 
                if ff:
                    torch.save(net.ff.state_dict(), outfile)
                else:
                    torch.save(net.state_dict(), outfile)
                print('----- SAVED MODEL -----')
                tally = 0
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
    
    print('Best Performance: Epoch {0:3d}, Loss {1:7.4f}, AUC {2:7.4f}'.format(best_n, overall_loss_tracker[best_n], overall_auc_tracker[best_n]))

    return overall_loss_tracker, overall_auc_tracker, y_tracker, y_prob_tracker

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
            
def run_training_epoch(epoch_num, train_loader, val_loader, net, criterion, optimizer, device, verbose=True, wait_time=1, max_batches=20, splits=['Train', 'Val']):
    total_loss = 0.0
    
    for t, ((x, y), (x_val, y_val)) in enumerate(zip(tqdm(train_loader), cycle(val_loader))):
        if max_batches != -1 and t >= max_batches:
            break
            
        else:
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
                    print(PRINT_STMT.format(epoch_num, t, loss.detach().cpu(), auc, loss_val, auc_val, *splits))

def run_validation_epoch(epoch_num, val_loader, net, criterion, device, verbose=True, max_batches=20, splits=['Val', 'CumVal']):
    net.eval()
    
    loss_tracker = np.array([])
    y_prob_tracker = np.array([])
    y_tracker = np.array([])

    for t, (x_val, y_val) in enumerate(tqdm(val_loader)):
        if max_batches != -1 and t >= max_batches:
            break
            
        else:
            with torch.no_grad():
                y_pred_val = net(x_val.to(device))
                loss_val = criterion(y_pred_val, y_val.to(device))
                loss_tracker = np.concatenate((loss_tracker, loss_val.detach().cpu().squeeze(-1).numpy()))

                y_prob_val = torch.sigmoid(y_pred_val.detach().cpu()).squeeze(-1).numpy()
                y_prob_tracker = np.concatenate((y_prob_tracker, y_prob_val))

                y_tracker = np.concatenate((y_tracker, y_val.squeeze(-1).numpy()))

                if verbose:
                    try:
                        auc_val = roc_auc_score(y_val.squeeze(-1).numpy(), y_prob_val)
                        auc_all = roc_auc_score(y_tracker, y_prob_tracker)
                    except:
                        auc_val = 0.0
                        auc_all = 0.0

                    loss_val = torch.mean(loss_val.detach().cpu())

                    print(PRINT_STMT.format(epoch_num, t, loss_val, auc_val, np.mean(loss_tracker), auc_all, *splits))

    return np.mean(loss_tracker), auc_all, y_tracker, y_prob_tracker