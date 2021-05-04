import os
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))
from caml.models import feedforward

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
from tqdm.contrib import tzip
from sklearn.metrics import roc_auc_score

import pdb

PRINT_STMT = 'Epoch {0:3d}, Task {1:3d}, {6:6} Loss {2:7.4f} AUC {3:7.4f}, {7:6} Loss {4:7.4f} AUC {5:7.4f}'


def init_models(hidden_size, output_size, n_local, device, dropout=0.0, resnet_file=None, maml_file=None, bias=True, freeze=True, pretrained=True):

    net = models.resnet18(pretrained=pretrained)
    embed_size = net.fc.weight.shape[1]
    net.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(embed_size, output_size, bias=bias))

    if resnet_file is not None and os.path.exists(resnet_file):
        saved_state = torch.load(resnet_file, map_location=lambda storage, loc: storage)
        net.load_state_dict(saved_state)

    net.fc = nn.Identity()
    net.to(device)

    if freeze:
        for param in net.parameters():
            param.requires_grad = False

    global_model = feedforward.FeedForwardNet(embed_size, hidden_size, output_size, dropout=dropout)
    
    if maml_file is not None and os.path.exists(maml_file):
        saved_state = torch.load(maml_file, map_location=lambda storage, loc: storage)
        global_model.load_state_dict(saved_state)
    
    global_model.to(device)

    global_theta = []
    for p in global_model.parameters():
        global_theta.append(p.detach().clone().to(device))

    local_models = []
    for i in range(n_local):
        local_models.append(feedforward.FeedForwardNet(embed_size, hidden_size, output_size, global_theta, dropout=dropout).to(device)) 

    return net, global_model, local_models, global_theta


def train_model(n_epochs, train_loaders, val_loaders, alpha, eta, wd, factor, net, global_model, local_models, global_theta, 
                criterions, device, n_steps, n_testtrain, n_testtest, patience, outfile, statsfile, n_choose=5, wait_time=1, 
                training=True, pool=None, batch_size=None, num_tiles=None, randomize=False, verbose=True, random_seed=31321):

    tally, best_n, best_auc, best_loss = 0, 0, 0, 1e9
    n_local = len(local_models)
    replace = n_choose > n_local

    if random_seed is not None:
        np.random.seed(random_seed)

    for n in tqdm(range(n_epochs)):
        if training:
            ts = np.random.choice(np.arange(n_local), n_choose, replace=replace)

            grads, local_models = run_local_train(n, ts, train_loaders, alpha, wd, net, local_models, global_theta, criterions[0], device, wait_time, 
                                                  pool=pool, batch_size=batch_size, num_tiles=num_tiles, randomize=randomize, verbose=verbose)
            
            global_theta, global_model = run_global_train(global_theta, global_model, grads, eta)
            
            for i in range(n_local):
                local_models[i].update_params(global_theta)
        
        #loss, auc, ys, yps = stats
        stats = run_validation(n, val_loaders, alpha, wd, net, global_model, global_theta, criterions, device, n_steps=n_steps, n_testtrain=n_testtrain, 
                               n_testtest=n_testtest, wait_time=wait_time, pool=pool, batch_size=batch_size, num_tiles=num_tiles, randomize=randomize, 
                               verbose=verbose)
        
        with open(statsfile, 'ab') as f:
            pickle.dump(stats, f) 

        loss = np.mean(stats[0])

        if training:
            if loss < best_loss: 
                torch.save(global_model.state_dict(), outfile)
                print('----- SAVED MODEL -----')
                tally = 0
            else:
                tally += 1

            if tally > patience:
                saved_state = torch.load(outfile, map_location=lambda storage, loc: storage)
                global_model.load_state_dict(saved_state)
                print('----- RELOADED MODEL ----- | Epoch {0:3d}, Loss {1:7.4f}, AUC {2:7.4f}'.format(best_n, best_loss, best_auc))

                alpha = max(factor * alpha, 1e-12)
                eta = max(factor * eta, 1e-12)
                print('-------- LR DECAY -------- | Alpha: {0:0.8f}, Eta: {1:0.8f}'.format(alpha, eta))

                tally = 0

        if loss < best_loss: 
            best_n = n
            best_loss = loss 
            best_auc = np.mean(stats[1])

    print('Best Performance: Epoch {0:3d}, Loss {1:7.4f}, AUC {2:7.4f}'.format(best_n, best_loss, best_auc))


def run_local_train(epoch_num, ts, train_loaders, alpha, wd, net, local_models, global_theta, criterion, device, wait_time, 
                    pool=None, batch_size=None, num_tiles=None, randomize=False, verbose=True, splits=['FwdOne', 'FwdTwo']):
    '''
    Note: 
    - Local training currently allows for only Adam optimizer
    '''

    net.eval()

    total_loss = 0.0
    
    if randomize:
        wait_time_orig = wait_time
        batch_size_orig = batch_size
        batch_size = 1
            
    grads = [torch.zeros(p.shape).to(device) for p in local_models[0].parameters()]

    for t in tqdm(ts):
        local_model = local_models[t]
        train_loader = train_loaders[t]

        optimizer = optim.Adam(local_model.parameters(), lr=alpha, weight_decay=wd)
        loss_tracker, auc_tracker, y_prob_tracker, y_tracker = [], [], [], []
        local_model.train()

        if randomize:
            wait_time = np.random.choice(np.arange(1, batch_size_orig * wait_time_orig + 1))

        for i, (x, y) in enumerate(train_loader):
            x = x[:batch_size, ]

            if pool is not None:
                x = x.to(device).contiguous().view(-1, x.shape[-3], x.shape[-2], x.shape[-1])
                y_pred = local_model(net(x))
                y_pred = y_pred.contiguous().view(batch_size, num_tiles, -1)
                y_pred = pool(y_pred, dim=1)
            else:
                y_pred = local_model(net(x.to(device)))
            
            loss = criterion(y_pred, y.to(device)) / wait_time
            loss.backward()

            total_loss += loss.detach().cpu().numpy()

            y_prob = torch.sigmoid(y_pred.detach().cpu())
            y_prob_tracker.append(y_prob.squeeze(-1).numpy())
            y_tracker.append(y.squeeze(-1).numpy())
                
            # first forward pass, update local params
            if i == wait_time - 1:
                optimizer.step()
                optimizer.zero_grad()

                loss_tracker.append(total_loss)
                total_loss = 0.0
                
            # second forward pass, store local grads
            if i == (wait_time * 2) - 1:
                grads[0] = grads[0] + local_model.lnr1.weight.grad.data
                grads[1] = grads[1] + local_model.lnr1.bias.grad.data
                grads[2] = grads[2] + local_model.lnr2.weight.grad.data
                grads[3] = grads[3] + local_model.lnr2.bias.grad.data

                local_model.update_params(global_theta)
                
                loss_tracker.append(total_loss)
                total_loss = 0.0
    
                if verbose:
                    bs = batch_size * wait_time
                    for j in range(2):
                        try:
                            auc_tracker.append(roc_auc_score(y_tracker[bs * j : bs * (j + 1)], y_prob_tracker[bs * j : bs * (j + 1)]))
                        except:
                            auc_tracker.append(0.0)

                    print(PRINT_STMT.format(epoch_num, t, loss_tracker[0], auc_tracker[0], loss_tracker[1], auc_tracker[1], *splits))
                
                break

    return grads, local_models


def run_global_train(global_theta, global_model, grads, eta):

    global_theta = [global_theta[i] - (eta * grads[i]) for i in range(len(global_theta))]

    global_model.update_params(global_theta)

    return global_theta, global_model


def run_validation(epoch_num, val_loaders, alpha, wd, net, global_model, global_theta, criterions, device, n_steps=1, n_testtrain=50, n_testtest=50, wait_time=1, 
                   pool=None, batch_size=None, num_tiles=None, randomize=False, verbose=True, splits=['TaskVal', 'CumVal']):

    net.eval()

    n_testtrain_orig = n_testtrain
        
    losses_tracker, loss_tracker, auc_tracker, y_prob_tracker, y_tracker = np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    for t, (metatrain_loader, metatest_loader) in enumerate(tzip(*val_loaders)):
    
        if randomize:
            n_testtrain = np.random.choice(np.arange(1, n_testtrain_orig + 1))   
            
            if n_testtrain % batch_size == 0:
                train_batch_size = batch_size
            else:
                train_batch_size = 1
                
            wait_time = max(min(len(metatrain_loader), n_testtrain // train_batch_size), 1) 
            
        else:
            train_batch_size = batch_size

        optimizer = optim.Adam(global_model.parameters(), lr=alpha, weight_decay=wd)

        criterion = criterions[0]

        global_model.train()
        
        # meta-test train
        for ns in range(n_steps):
            for i, (x, y) in enumerate(metatrain_loader):    
                x = x[:train_batch_size, ]
                y = y[:train_batch_size, ]

                if pool is not None:
                    x = x.to(device).contiguous().view(-1, x.shape[-3], x.shape[-2], x.shape[-1])
                    y_pred = global_model(net(x))
                    y_pred = y_pred.contiguous().view(train_batch_size, num_tiles, -1)
                    y_pred = pool(y_pred, dim=1)
                else:
                    y_pred = global_model(net(x.to(device)))
                
                loss = criterion(y_pred, y.to(device)) / wait_time
                loss.backward()

                if i == wait_time - 1:
                    optimizer.step()
                    optimizer.zero_grad()
                    break
        
        criterion = criterions[1]
        
        global_model.eval()

        bs = 0
        
        # meta-test test
        for i, (x, y) in enumerate(metatest_loader):
            val_batch_size = x.shape[0]
            bs += val_batch_size
            
            with torch.no_grad():
                if pool is not None:
                    x = x.to(device).contiguous().view(-1, x.shape[-3], x.shape[-2], x.shape[-1])
                    y_pred = global_model(net(x))
                    y_pred = y_pred.contiguous().view(val_batch_size, num_tiles, -1)
                    y_pred = pool(y_pred, dim=1)
                else:
                    y_pred = global_model(net(x.to(device)))

            loss = criterion(y_pred, y.to(device))
            losses_tracker = np.concatenate((losses_tracker, loss.cpu().squeeze(-1).numpy()))

            y_prob = torch.sigmoid(y_pred.cpu())    
            y_prob_tracker = np.concatenate((y_prob_tracker, y_prob.squeeze(-1).numpy()))
            y_tracker = np.concatenate((y_tracker, y.squeeze(-1).numpy()))

        try:
            auc = roc_auc_score(y_tracker[-bs:], y_prob_tracker[-bs:])
        except:
            auc = 0.0

        auc_tracker = np.concatenate((auc_tracker, [auc]))
        loss_tracker = np.concatenate((loss_tracker, [np.mean(losses_tracker[-bs:])]))

        if verbose:
            try:
                auc_all = roc_auc_score(y_tracker, y_prob_tracker)
            except:
                auc_all = 0.0

            print(PRINT_STMT.format(epoch_num, t, np.mean(loss_tracker[-bs:]), auc, np.mean(loss_tracker), auc_all, *splits))

        global_model.update_params(global_theta)

    return loss_tracker, auc_tracker, y_tracker, y_prob_tracker