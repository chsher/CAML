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

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

PRINT_STMT = 'Epoch {0:3d}, Task {1:3d}, {6:6} Loss {2:7.4f} AUC {3:7.4f}, {7:6} Loss {4:7.4f} AUC {5:7.4f}'

def init_models(new_hidden_size, output_size, n_local, device, dropout=0.0, resnet_file=None, maml_file=None):
    net = models.resnet18(pretrained=False)
    hidden_size = net.fc.weight.shape[1]
    net.fc = nn.Linear(hidden_size, output_size, bias=True)

    if resnet_file is not None:
        saved_state = torch.load(resnet_file, map_location=lambda storage, loc: storage)
        net.load_state_dict(saved_state)

    net.fc = nn.Identity()
    net.to(device)

    for param in net.parameters():
        param.requires_grad = False

    global_model = feedforward.FeedForwardNet(hidden_size, new_hidden_size, output_size, dropout=dropout)
    
    if maml_file is not None:
        saved_state = torch.load(maml_file, map_location=lambda storage, loc: storage)
        global_model.load_state_dict(saved_state)
    
    global_model.to(device)

    global_theta = []
    for p in global_model.parameters():
        global_theta.append(p.detach().clone().to(device))

    local_models = []
    for i in range(n_local):
        local_models.append(feedforward.FeedForwardNet(hidden_size, new_hidden_size, output_size, global_theta, dropout=dropout).to(device)) 

    return net, global_model, local_models, global_theta

def train_model(n_epochs, train_loaders, val_loaders, alpha, eta, wd, factor, net, global_model, local_models, global_theta, criterions, device, n_steps, patience, outfile, n_choose=5, verbose=True, random_seed=31321):

    tally = 0
    best_n = 0
    old_loss = 1e9
    n_local = len(local_models)

    overall_loss_tracker = []
    overall_auc_tracker = []
    y_tracker = []
    y_prob_tracker = []

    if random_seed is not None:
        np.random.seed(random_seed)
    
    n_tasks = len(local_models)
    if n_choose > n_tasks:
        replace=True
    else:
        replace=False

    for n in tqdm(range(n_epochs)):
        ts = np.random.choice(np.arange(n_tasks), n_choose, replace=replace)

        grads, local_models = run_local_train(n, ts, train_loaders, alpha, wd, net, local_models, criterions[0], device, verbose)
        global_theta, global_model = run_global_train(global_theta, global_model, grads, eta)
        for i in range(n_local):
            local_models[i].update_params(global_theta)
        
        loss, auc, ys, yps = run_validation(n, val_loaders, alpha, wd, net, global_model, global_theta, criterions, device, n_steps, verbose)
        overall_loss_tracker.append(loss)
        overall_auc_tracker.append(auc)
        y_tracker.append(ys)
        y_prob_tracker.append(yps)
        
        if loss < old_loss: 
            best_n = n
            old_loss = loss 
            torch.save(global_model.state_dict(), outfile)
            print('----- SAVED MODEL -----')
        else:
            tally += 1

        if tally > patience:
            saved_state = torch.load(outfile, map_location=lambda storage, loc: storage)
            global_model.load_state_dict(saved_state)
            print('----- RELOADED MODEL -----')

            alpha = factor * alpha
            eta = factor * eta
            print('----- LR DECAY ----- | Alpha: {0:0.8f}, Eta: {1:0.8f}'.format(alpha, eta))
            
            tally = 0
    
    print('Best Performance: Epoch {0:3d}, Loss {1:7.4f}, AUC {2:7.4f}'.format(best_n, overall_loss_tracker[best_n], overall_auc_tracker[best_n]))
    return overall_loss_tracker, overall_auc_tracker, y_tracker, y_prob_tracker

def run_local_train(epoch_num, ts, train_loaders, alpha, wd, net, local_models, criterion, device, verbose=True, splits=['FwdOne', 'FwdTwo']):
    '''
    Note: 
    - currently only allows for Adam optimizer
    '''
    net.eval()

    grads = [torch.zeros(p.shape).to(device) for p in local_models[0].parameters()]

    for t in ts:
        local_model = local_models[t]
        local_model.train()

        train_loader = train_loaders[t]

        for i, (x, y) in enumerate(train_loader):
            # first forward pass, update local params
            if i == 0:
                optimizer = torch.optim.Adam(local_model.parameters(), lr=alpha, weight_decay=wd)

                y_pred = local_model(net(x.to(device)))

                loss = criterion(y_pred, y.to(device))
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()
            
            # second forward pass, store grads
            elif i == 1:
                y_pred = local_model(net(x.to(device)))

                loss = criterion(y_pred, y.to(device))
                loss.backward()

                grads[0] = grads[0] + local_model.lnr1.weight.grad.data
                grads[1] = grads[1] + local_model.lnr1.bias.grad.data
                grads[2] = grads[2] + local_model.lnr2.weight.grad.data
                grads[3] = grads[3] + local_model.lnr2.bias.grad.data

            else:
                break
        
    if verbose:
        y_prob = torch.sigmoid(y_pred.detach().cpu())
        try:
            auc = roc_auc_score(y.squeeze(-1).numpy(), y_prob.squeeze(-1).numpy())
        except:
            auc = 0.0

        print(PRINT_STMT.format(epoch_num, t, 0.0, 0.0, loss.detach().cpu(), auc, *splits))

    return grads, local_models

def run_global_train(global_theta, global_model, grads, eta):
    global_theta = [global_theta[i] - (eta * grads[i]) for i in range(len(global_theta))]

    global_model.update_params(global_theta)

    return global_theta, global_model

def run_validation(epoch_num, val_loaders, alpha, wd, net, global_model, global_theta, criterions, device, n_steps=1, verbose=True, splits=['Val', 'CumVal']):
    net.eval()
    global_model.eval()

    loss_tracker = np.array([])
    y_tracker = np.array([])
    y_prob_tracker = np.array([])

    for t, val_loader in enumerate(tqdm(val_loaders)):
        criterion = criterions[0]
        global_model.update_params(global_theta)
        optimizer = torch.optim.Adam(global_model.parameters(), lr=alpha, weight_decay=wd)

        for i, (x, y) in enumerate(val_loader):
            if i < n_steps:
                y_pred = global_model(net(x.to(device)))
                loss = criterion(y_pred, y.to(device))

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            elif i >= n_steps:
                with torch.no_grad():
                    criterion = criterions[1]

                    y_pred = global_model(net(x.to(device)))
                    loss = criterion(y_pred, y.to(device))
                    loss_tracker = np.concatenate((loss_tracker, loss.detach().cpu().squeeze(-1).numpy()))
                    
                    y_prob = torch.sigmoid(y_pred.detach().cpu()).squeeze(-1).numpy()    
                    y_prob_tracker = np.concatenate((y_prob_tracker, y_prob))
                    
                    y_tracker = np.concatenate((y_tracker, y.squeeze(-1).numpy()))

                    if verbose:
                        try:
                            auc = roc_auc_score(y.squeeze(-1).numpy(), y_prob)
                            auc_all = roc_auc_score(y_tracker, y_prob_tracker)
                        except:
                            auc = 0.0
                            auc_all = 0.0

                        loss = torch.mean(loss.detach().cpu())
                        
                        print(PRINT_STMT.format(epoch_num, t, loss, auc, np.mean(loss_tracker), auc_all, *splits))

                break

    return np.mean(loss_tracker), auc_all, y_tracker, y_prob_tracker