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
from sklearn.metrics import roc_auc_score

PRINT_STMT = 'Epoch {0:3d}, Task {1:3d}, {6:6} Loss {2:7.4f} AUC {3:7.4f}, {7:6} Loss {4:7.4f} AUC {5:7.4f}'

def init_models(hidden_size, output_size, n_local, device, dropout=0.0, resnet_file=None, maml_file=None, freeze=True, bias=True):
    net = models.resnet18(pretrained=True)
    embed_size = net.fc.weight.shape[1]
    net.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(embed_size, output_size, bias=bias))
    #net.fc = nn.Linear(embed_size, output_size, bias=True)

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
                training=True, pool=None, batch_size=None, num_tiles=None, verbose=True, random_seed=31321):

    tally = 0
    best_n, best_auc, old_loss = 0, 0, 1e9
    n_local = len(local_models)

    #overall_loss_tracker = []
    #overall_auc_tracker = []
    #y_tracker = []
    #y_prob_tracker = []

    if random_seed is not None:
        np.random.seed(random_seed)
    
    if n_choose > n_local:
        replace = True
    else:
        replace = False

    for n in tqdm(range(n_epochs)):
        if training:
            ts = np.random.choice(np.arange(n_local), n_choose, replace=replace)

            grads, local_models = run_local_train(n, ts, train_loaders, alpha, wd, net, local_models, global_theta, criterions[0], device, wait_time, 
                                                  pool, batch_size, num_tiles, verbose)
            
            global_theta, global_model = run_global_train(global_theta, global_model, grads, eta)
            
            for i in range(n_local):
                local_models[i].update_params(global_theta)
        
        stats = run_validation(n, val_loaders, alpha, wd, net, global_model, global_theta, criterions, device, n_steps, n_testtrain, n_testtest, wait_time, 
                               pool, batch_size, num_tiles, verbose)
        
        #loss, auc, ys, yps
        #overall_loss_tracker.append(loss)
        #overall_auc_tracker.append(auc)
        #y_tracker.append(ys)
        #y_prob_tracker.append(yps)
        
        with open(statsfile, 'ab') as f:
            pickle.dump(stats, f) 

        loss = np.mean(stats[0])

        if training:
            if loss < old_loss: 
                torch.save(global_model.state_dict(), outfile)
                print('----- SAVED MODEL -----')
                tally = 0
            else:
                tally += 1

            if tally > patience:
                saved_state = torch.load(outfile, map_location=lambda storage, loc: storage)
                global_model.load_state_dict(saved_state)
                print('----- RELOADED MODEL -----')

                alpha = max(factor * alpha, 1e-12)
                eta = max(factor * eta, 1e-12)
                print('----- LR DECAY ----- | Alpha: {0:0.8f}, Eta: {1:0.8f}'.format(alpha, eta))

                tally = 0

        if loss < old_loss: 
            best_n = n
            best_loss = loss 
            best_auc = np.mean(stats[1])
    
    #print('Best Performance: Epoch {0:3d}, Loss {1:7.4f}, AUC {2:7.4f}'.format(best_n, overall_loss_tracker[best_n], overall_auc_tracker[best_n]))
    print('Best Performance: Epoch {0:3d}, Loss {1:7.4f}, AUC {2:7.4f}'.format(best_n, best_loss, best_auc))

    #return overall_loss_tracker, overall_auc_tracker, y_tracker, y_prob_tracker

def run_local_train(epoch_num, ts, train_loaders, alpha, wd, net, local_models, global_theta, criterion, device, wait_time, 
                    pool=None, batch_size=None, num_tiles=None, verbose=True, splits=['FwdOne', 'FwdTwo']):
    '''
    Note: 
    - currently only allows for Adam optimizer
    '''
    net.eval()

    grads = [torch.zeros(p.shape).to(device) for p in local_models[0].parameters()]

    for t in tqdm(ts):
        local_model = local_models[t]
        optimizer = optim.Adam(local_model.parameters(), lr=alpha, weight_decay=wd)
        local_model.train()

        train_loader = train_loaders[t]

        total_loss = 0.0
        loss_tracker, auc_tracker, y_prob_tracker, y_tracker = np.array([]), np.array([]), np.array([]), np.array([])
        
        for i, (x, y) in enumerate(train_loader):
            # first forward pass, update local params
            if i < wait_time:
                if pool is not None:
                    x = x.to(device).contiguous().view(-1, x.shape[-3], x.shape[-2], x.shape[-1])
                    y_pred = local_model(net(x))
                    y_pred = y_pred.contiguous().view(batch_size, num_tiles, -1)
                    y_pred = pool(y_pred, dim=1)
                else:
                    y_pred = local_model(net(x.to(device)))
                
                loss = criterion(y_pred, y.to(device))
                total_loss += loss / wait_time

                y_prob = torch.sigmoid(y_pred.detach().cpu()).squeeze(-1).numpy() 
                y_prob_tracker = np.concatenate((y_prob_tracker, y_prob))
                y_tracker = np.concatenate((y_tracker, y.squeeze(-1).numpy()))
                    
                if i == wait_time - 1:
                    total_loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    loss_tracker = np.concatenate((loss_tracker, [total_loss.detach().cpu().numpy()]))
                    total_loss = 0.0
                
            # second forward pass, store grads
            elif (i >= wait_time) and (i < wait_time * 2):
                if pool is not None:
                    x = x.to(device).contiguous().view(-1, x.shape[-3], x.shape[-2], x.shape[-1])
                    y_pred = local_model(net(x))
                    y_pred = y_pred.contiguous().view(batch_size, num_tiles, -1)
                    y_pred = pool(y_pred, dim=1)
                else:
                    y_pred = local_model(net(x.to(device)))

                loss = criterion(y_pred, y.to(device))
                total_loss += loss / wait_time

                y_prob = torch.sigmoid(y_pred.detach().cpu()).squeeze(-1).numpy() 
                y_prob_tracker = np.concatenate((y_prob_tracker, y_prob))
                y_tracker = np.concatenate((y_tracker, y.squeeze(-1).numpy()))
                
                if i == (wait_time * 2) - 1:
                    total_loss.backward()

                    grads[0] = grads[0] + local_model.lnr1.weight.grad.data
                    grads[1] = grads[1] + local_model.lnr1.bias.grad.data
                    grads[2] = grads[2] + local_model.lnr2.weight.grad.data
                    grads[3] = grads[3] + local_model.lnr2.bias.grad.data

                    local_model.update_params(global_theta)
                    
                    loss_tracker = np.concatenate((loss_tracker, [total_loss.detach().cpu().numpy()]))
        
                    if verbose:
                        bs = wait_time * batch_size
                    
                        try:
                            auc_one = roc_auc_score(y_tracker[:bs], y_prob_tracker[:bs])
                        except:
                            auc_one = 0.0

                        try:
                            auc_two = roc_auc_score(y_tracker[bs:], y_prob_tracker[bs:])
                        except:
                            auc_two = 0.0

                        print(PRINT_STMT.format(epoch_num, t, loss_tracker[0], auc_one, loss_tracker[1], auc_two, *splits))
                    
                    break

    return grads, local_models

def run_global_train(global_theta, global_model, grads, eta):
    global_theta = [global_theta[i] - (eta * grads[i]) for i in range(len(global_theta))]

    global_model.update_params(global_theta)

    return global_theta, global_model

def run_validation(epoch_num, val_loaders, alpha, wd, net, global_model, global_theta, criterions, device, n_steps=1, n_testtrain=50, n_testtest=50, wait_time=1, 
                   pool=None, batch_size=None, num_tiles=None, verbose=True, splits=['TaskVal', 'CumVal']):
    net.eval()
    
    val_wait_time = max(n_testtest // batch_size, 1)

    loss_tracker, auc_tracker, y_prob_tracker, y_tracker = np.array([]), np.array([]), np.array([]), np.array([])

    for t, val_loader in enumerate(tqdm(val_loaders)):
        global_model.update_params(global_theta)
        optimizer = optim.Adam(global_model.parameters(), lr=alpha, weight_decay=wd)
        global_model.train()
        
        criterion = criterions[0]
        total_loss = 0.0

        for ns in range(n_steps):
            for i, (x, y) in enumerate(val_loader):
                if i < wait_time:      
                    if x.shape[0] > n_testtrain:
                        x = x[:n_testtrain, ]
                        y = y[:n_testtrain, ]

                    if pool is not None:
                        x = x.to(device).contiguous().view(-1, x.shape[-3], x.shape[-2], x.shape[-1])
                        y_pred = global_model(net(x))
                        y_pred = y_pred.contiguous().view(batch_size, num_tiles, -1)
                        y_pred = pool(y_pred, dim=1)
                    else:
                        y_pred = global_model(net(x.to(device)))
                    
                    loss = criterion(y_pred, y.to(device))
                    total_loss += loss / wait_time

                if i == wait_time - 1:
                    total_loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    total_loss = 0.0

                    break

        global_model.eval()
        criterion = criterions[1]
        total_loss = 0.0

        for i, (x, y) in enumerate(val_loader):
            if (i >= wait_time) and (i < wait_time + val_wait_time):
                if x.shape[0] > n_testtest:
                    x = x[:n_testtest, ]
                    y = y[:n_testtest, ]

                with torch.no_grad():
                    if pool is not None:
                        x = x.to(device).contiguous().view(-1, x.shape[-3], x.shape[-2], x.shape[-1])
                        y_pred = global_model(net(x))
                        y_pred = y_pred.contiguous().view(batch_size, num_tiles, -1)
                        y_pred = pool(y_pred, dim=1)
                    else:
                        y_pred = global_model(net(x.to(device)))
                        
                    loss = criterion(y_pred, y.to(device))
                    total_loss += torch.mean(loss.detach().cpu()) / val_wait_time
                    loss_tracker = np.concatenate((loss_tracker, loss.detach().cpu().squeeze(-1).numpy()))
                    
                    y_prob = torch.sigmoid(y_pred.detach().cpu()).squeeze(-1).numpy()    
                    y_prob_tracker = np.concatenate((y_prob_tracker, y_prob))
                    y_tracker = np.concatenate((y_tracker, y.squeeze(-1).numpy()))

                if i == (wait_time + val_wait_time) - 1:
                    try:
                        bs = val_wait_time * min(batch_size, n_testtest)
                        auc = roc_auc_score(y_tracker[-bs:], y_prob_tracker[-bs:])
                        #auc = roc_auc_score(y.squeeze(-1).numpy(), y_prob)
                    except:
                        auc = 0.0

                    try:
                        auc_all = roc_auc_score(y_tracker, y_prob_tracker)
                    except:
                        auc_all = 0.0

                    auc_tracker = np.concatenate((auc_tracker, [auc]))

                    if verbose:
                        print(PRINT_STMT.format(epoch_num, t, total_loss, auc, np.mean(loss_tracker), auc_all, *splits))

                    break

    global_model.update_params(global_theta)

    return np.mean(np.vstack(np.split(loss_tracker, len(val_loaders))), axis=1), auc_tracker, y_tracker, y_prob_tracker