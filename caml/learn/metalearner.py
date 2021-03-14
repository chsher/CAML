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

PRINT_STMT = 'Epoch {0:3d}, Minibatch {1:3d}, {6:6} Loss {2:7.4f} AUC {3:7.4f}, {7:6} Loss {4:7.4f} AUC {5:7.4f}'

def init_models(new_hidden_size, output_size, n_local, device, dropout=0.0, resnet_file=None, maml_file=None):
    net = models.resnet18(pretrained=False)
    hidden_size = net.fc.weight.shape[1]
    net.fc = nn.Linear(hidden_size, output_size, bias=True)

    if resnet_file is not None
        saved_state = torch.load(state_dict_file, map_location=lambda storage, loc: storage)
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

def train_model(n_epochs, train_loader, val_loader, alpha, eta, wd, factor, net, global_model, local_models, theta_global, criterions, device, patience, outfile, verbose=True):
    tally = 0
    old_loss = 1e9
    n_local = len(local_models)
    overall_loss_tracker = []
    overall_auc_tracker = []
    
    for n in tqdm(range(n_epochs)):
        for t, (x, y) in enumerate(train_loader):
            grads, local_models = run_local_train(n, t, x, y, alpha, wd, net, local_models, criterions[0], device, verbose)
            global_theta, global_model = run_global_train(global_theta, global_model, grads, eta)
            for i in range(n_local):
                local_models[i].update_params(global_theta)
            
        loss, auc = run_validation_epoch(n, val_loader, net, global_model, criterions[1], device, verbose)
        overall_loss_tracker.append(loss)
        overall_auc_tracker.append(auc)
        
        if loss < old_loss: 
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
            
    return overall_loss_tracker, overall_auc_tracker

def run_local_train(epoch_num, step_num, x, y, alpha, wd, net, local_models, criterion, device, verbose=True, splits=['FwdOne', 'FwdTwo']):
    '''
    Note: 
    - currently only allows for Adam optimizer
    '''
    net.eval()
    idx = int(x.shape[0] // 2)
    num_tasks = int(x.shape[1])

    grads = [torch.zeros(p.shape).to(device) for p in local_models[0].parameters()]

    for t in range(num_tasks):
        local_model = local_models[t]
        local_model.train()

        # first forward pass, update local params
        optimizer = torch.optim.Adam(local_model.parameters(), lr=alpha, weight_decay=wd)

        inputs = x[:idx, t, :, :, :].to(device)
        embed = net(inputs)
        y_pred1 = local_model(embed)

        loss1 = criterion(y_pred1, y[:idx, t].unsqueeze(1).to(device))
        loss1.backward()

        optimizer.step()
        optimizer.zero_grad()
        
        # second forward pass, store grads
        inputs = x[idx:, t, :, :, :].to(device)
        embed = net(inputs)
        y_pred2 = local_model(embed)

        loss2 = criterion(y_pred2, y[idx:, t].unsqueeze(1).to(device))
        loss2.backward()

        grads[0] = grads[0] + local_model.linear1.weight.grad.data
        grads[1] = grads[1] + local_model.linear1.bias.grad.data
        grads[2] = grads[2] + local_model.linear2.weight.grad.data
        grads[3] = grads[3] + local_model.linear2.bias.grad.data

        optimizer.zero_grad()
        
    if verbose:
        y_prob1 = torch.sigmoid(y_pred1.detach().cpu())
        y_prob2 = torch.sigmoid(y_pred2.detach().cpu())
        try:
            auc1 = roc_auc_score(y[:idx, t].numpy(), y_prob1.squeeze(-1).numpy())
            auc2 = roc_auc_score(y[idx:, t].numpy(), y_prob2.squeeze(-1).numpy())
        except:
            auc1 = 0.0
            auc2 = 0.0

        print(PRINT_STMT.format(epoch_num, step_num, loss1.item(), auc1, loss2.item(), auc2, *splits))

    return grads, local_models

def run_global_train(global_theta, global_model, grads, eta):
    global_theta = [global_theta[i] - (eta * grads[i]) for i in range(len(global_theta))]

    global_model.update_params(global_theta)

    return global_theta, global_model

def run_validation_epoch(epoch_num, val_loader, net, global_model, criterions[1], device, verbose=True, splits=['Val', 'CumVal']):
    net.eval()
    global_model.eval()

    loss_tracker = torch.FloatTensor()
    y_tracker = np.array([])
    y_prob_tracker = np.array([])

    for t, (x_val, y_val) in enumerate(tqdm(val_loader)):
        with torch.no_grad():
            x_val = x_val.reshape(-1, x_val.shape[-3], x_val.shape[-2], x_val.shape[-1])
            embed_val = net(x_val.to(device))
            y_pred_val = global_model(embed_val)

            loss_val = criterion(y_pred_val, y_val.to(device))
            loss_tracker = torch.cat((loss_tracker, loss_val.detach().cpu()))
            
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
            print(PRINT_STMT.format(epoch_num, t, loss_val.item(), auc_val, torch.mean(loss_tracker), auc_all, *splits))

    return torch.mean(loss_tracker).item(), auc_all