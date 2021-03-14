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

def train_model(n_epochs, train_loader, val_loader, net, criterions, optimizer, device, scheduler, patience, outfile, verbose=True):
    tally = 0
    old_loss = 1e9
    overall_loss_tracker = []
    overall_auc_tracker = []
    
    for n in tqdm(range(n_epochs)):
        run_training_epoch(n, train_loader, val_loader, net, criterions[0], optimizer, device, verbose)
        
        loss, auc = run_validation_epoch(n, val_loader, net, criterions[1], device, verbose)
        overall_loss_tracker.append(loss)
        overall_auc_tracker.append(auc)
        
        if loss < old_loss: 
            old_loss = loss 
            torch.save(net.state_dict(), outfile)
            print('----- SAVED MODEL -----')
        else:
            tally += 1
                
        if scheduler is not None:
            scheduler.step(loss)

        if tally > patience:
            saved_state = torch.load(outfile, map_location=lambda storage, loc: storage)
            net.load_state_dict(saved_state)
            print('----- RELOADED MODEL -----')
            tally = 0
            
    return overall_loss_tracker, overall_auc_tracker

def cycle(iterable):
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
        
def run_validation_epoch(epoch_num, val_loader, net, criterion, device, verbose=True, splits=['Val', 'CumVal']):
    net.eval()
    loss_tracker = torch.FloatTensor()
    y_tracker = np.array([])
    y_prob_tracker = np.array([])

    for t, (x_val, y_val) in enumerate(tqdm(val_loader)):
        with torch.no_grad():
            y_pred_val = net(x_val.to(device))
            loss_val = criterion(y_pred_val, y_val.to(device))
            loss_tracker = torch.cat((loss_tracker, loss_val.detach().cpu()))
            
            y_prob_val = torch.sigmoid(y_pred_val.detach().cpu()).squeeze(-1).numpy()
                
            y_tracker = np.concatenate((y_tracker, y_val.squeeze(-1).numpy()))
            y_prob_tracker = np.concatenate((y_prob_tracker, y_prob_val))
        
        if verbose:
            try:
                auc_val = roc_auc_score(y_val.squeeze(-1).numpy(), y_prob_val)
                auc_all = roc_auc_score(y_tracker, y_prob_tracker)
            except:
                auc_val = 0.0
                auc_all = 0.0
            loss_val = torch.mean(loss_val.detach().cpu())
            print(PRINT_STMT.format(epoch_num, t, loss_val, auc_val, torch.mean(loss_tracker), auc_all, *splits))

    return torch.mean(loss_tracker).item(), auc_all