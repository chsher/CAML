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

import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

import pdb

PRINT_STMT = 'Epoch {0:3d}, Minibatch {1:3d}, {6:6} Loss {2:7.4f} AUC {3:7.4f}, {7:6} Loss {4:7.4f} AUC {5:7.4f}'


def train_model(n_epochs, train_loader, val_loaders, net, criterions, optimizer, device, scheduler, patience,outfile, statsfile, resfile_new=None, 
    n_steps=1, n_testtrain=50, wait_time=1, pool=None, batch_size=1, num_tiles=50, max_batches=[20, 20], grad_adapt=False, ff=False, freeze=True, 
    training=True, test_loaders=None, verbose=True):

    tally, best_n, best_auc, best_loss = 0, 0, 0, 1e9

    for n in range(n_epochs):
        if training:
            if grad_adapt:
                for ns in tqdm(range(n_steps)):
                    #verbose = True if ns == n_steps - 1 else False
                    run_training_epoch(n, train_loader, val_loaders[0][0][0], net, criterions[0], optimizer, device, 
                                       wait_time=wait_time, max_batches=max_batches[0], verbose=verbose)
            else:    
                run_training_epoch(n, train_loader, val_loaders[0], net, criterions[0], optimizer, device, 
                                   wait_time=wait_time, max_batches=max_batches[0], verbose=verbose)
        
        #loss, auc, ys, yps = stats
        if grad_adapt:
            alpha = optimizer.param_groups[0]['lr']
            wd = optimizer.param_groups[0]['weight_decay']

            global_theta = []
            for p in net.ff.parameters():
                global_theta.append(p.detach().clone().to(device))

            stats = metalearner.run_validation(n, val_loaders, alpha, wd, net.resnet, net.ff, global_theta, criterions, device, 
                                               n_steps=n_steps, wait_time=wait_time, pool=pool, batch_size=batch_size, num_tiles=num_tiles, 
                                               max_batches=max_batches[-1], verbose=verbose)
        else:
            stats = run_validation_epoch(n, val_loaders[0], net, criterions[1], device, wait_time=wait_time, max_batches=max_batches[1], verbose=verbose)
            
        with open(statsfile, 'ab') as f:
            pickle.dump(stats, f)   

        loss = np.mean(stats[0]) if grad_adapt else stats[0]
        
        if training:
            if loss < best_loss: 
                if ff:
                    torch.save(net.ff.state_dict(), outfile)
                    #if not freeze:
                    #    torch.save(net.resnet.state_dict(), resfile_new)
                else:
                    torch.save(net.resnet.state_dict(), outfile)
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
                    #if not freeze:
                    #    saved_state_new = torch.load(resfile_new, map_location=lambda storage, loc: storage)
                    #    net.resnet.load_state_dict(saved_state_new)
                else:
                    net.resnet.load_state_dict(saved_state)
                print('----- RELOADED MODEL ----- | Epoch {0:3d}, Loss {1:7.4f}, AUC {2:7.4f}'.format(best_n, best_loss, best_auc))
                tally = 0
                
        if loss < best_loss: 
            best_n = n
            best_loss = loss 
            best_auc = np.nanmean(stats[1]) if grad_adapt else stats[1]
    
    print('Best Val Performance: Epoch {0:3d}, Loss {1:7.4f}, AUC {2:7.4f}'.format(best_n, best_loss, best_auc))

    if test_loaders is not None:
        if grad_adapt:
            alpha = optimizer.param_groups[0]['lr']
            wd = optimizer.param_groups[0]['weight_decay']

            global_theta = []
            for p in net.ff.parameters():
                global_theta.append(p.detach().clone().to(device))

            stats = metalearner.run_validation(n, test_loaders, alpha, wd, net.resnet, net.ff, global_theta, criterions, device, 
                                               n_steps=n_steps, wait_time=wait_time, pool=pool, batch_size=batch_size, num_tiles=num_tiles, 
                                               max_batches=max_batches[-1], verbose=verbose)
        else:
            stats = run_validation_epoch(n, test_loaders[0], net, criterions[1], device, wait_time=wait_time, max_batches=max_batches[1], verbose=verbose)
            
        with open(statsfile, 'ab') as f:
            pickle.dump(stats, f)

        loss = np.mean(stats[0]) if grad_adapt else stats[0]
        auc = np.nanmean(stats[1]) if grad_adapt else stats[1]

        print('Held-Out Test Performance: Loss {0:7.4f}, AUC {:7.4f}'.format(loss, auc))

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
            

def run_training_epoch(epoch_num, train_loader, val_loader, net, criterion, optimizer, device, wait_time=1, max_batches=20, verbose=True, splits=['Train', 'Val']):
    
    total_loss, y_prob_tracker, y_tracker = 0.0, np.array([]), np.array([])
    total_loss_val, y_prob_tracker_val, y_tracker_val = 0.0, np.array([]), np.array([])
    
    for t, ((x, y), (x_val, y_val)) in enumerate(zip(tqdm(train_loader), cycle(val_loader))):
        if t >= max_batches and max_batches != -1:
            break

        elif t == (max_batches // wait_time) * wait_time and max_batches % wait_time > 0 and max_batches != -1:
            wait_time = max_batches % wait_time
            
        elif t == (len(train_loader) // wait_time) * wait_time and len(train_loader) % wait_time > 0 and max_batches == -1:
            wait_time = len(train_loader) % wait_time

        net.train()
        optimizer.zero_grad()

        y_pred = net(x.to(device))
        
        loss = criterion(y_pred, y.to(device)) / wait_time
        loss.backward()

        total_loss += loss.detach().cpu() 
        
        y_prob = torch.sigmoid(y_pred.detach().cpu())
        y_prob_tracker = np.concatenate((y_prob_tracker, y_prob.squeeze(-1).numpy()))
        y_tracker = np.concatenate((y_tracker, y.squeeze(-1).numpy()))
        
        net.eval()
        with torch.no_grad():
            y_pred_val = net(x_val.to(device))

        loss_val = criterion(y_pred_val, y_val.to(device)) / wait_time

        total_loss_val += loss_val.cpu() 
        
        y_prob_val = torch.sigmoid(y_pred_val.cpu())
        y_prob_tracker_val = np.concatenate((y_prob_tracker_val, y_prob_val.squeeze(-1).numpy()))
        y_tracker_val = np.concatenate((y_tracker_val, y_val.squeeze(-1).numpy()))

        if (t + 1) % wait_time == 0:
            net.train()
            optimizer.step()
            
            if verbose:
                try:
                    auc = roc_auc_score(y_tracker, y_prob_tracker)
                except:
                    auc = np.nan

                try:
                    auc_val = roc_auc_score(y_tracker_val, y_prob_tracker_val)
                except:
                    auc_val = np.nan

                print(PRINT_STMT.format(epoch_num, t, total_loss, auc, total_loss_val, auc_val, *splits))
                
            total_loss = 0.0
            total_loss_val = 0.0


def run_validation_epoch(epoch_num, val_loader, net, criterion, device, wait_time=1, max_batches=20, verbose=True, splits=['Val', 'CumVal']):

    net.eval()
    
    loss_tracker, y_prob_tracker, y_tracker = np.array([]), np.array([]), np.array([])

    for t, (x_val, y_val) in enumerate(tqdm(val_loader)):
        if t >= max_batches and max_batches != -1:
            break
            
        with torch.no_grad():
            y_pred_val = net(x_val.to(device))

        loss_val = criterion(y_pred_val, y_val.to(device)) 
        loss_tracker = np.concatenate((loss_tracker, loss_val.cpu().squeeze(-1).numpy()))

        y_prob_val = torch.sigmoid(y_pred_val.cpu())
        y_prob_tracker = np.concatenate((y_prob_tracker, y_prob_val.squeeze(-1).numpy()))
        y_tracker = np.concatenate((y_tracker, y_val.squeeze(-1).numpy()))

        if ((t + 1) % wait_time == 0) or (t == max_batches - 1) or (t == len(val_loader) - 1):
            try:
                auc_all = roc_auc_score(y_tracker, y_prob_tracker)
            except:
                auc_all = np.nan

            if verbose:
                try:
                    auc_val = roc_auc_score(y_val.squeeze(-1).numpy(), y_prob_val.squeeze(-1).numpy())
                except:
                    auc_val = np.nan

                print(PRINT_STMT.format(epoch_num, t, np.mean(loss_val.cpu().squeeze(-1).numpy()), auc_val, np.mean(loss_tracker), auc_all, *splits))

    return np.mean(loss_tracker), auc_all, y_tracker, y_prob_tracker
