import os
import torch
import torch.nn as nn
from torchvision import models

class FeedForwardNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, initial_vals=None, dropout=0.0, bias=True):
        super(FeedForwardNet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.initial_vals = initial_vals
        self.dropout = dropout
        self.bias = bias

        self.lnr1 = nn.Linear(self.input_size, self.hidden_size, bias=self.bias)
        self.lnr2 = nn.Linear(self.hidden_size, self.output_size, bias=self.bias)
        self.d = nn.Dropout(self.dropout)
        self.m = nn.Tanh()
        
        if self.initial_vals is not None:
            self.update_params(self.initial_vals)
        
    def forward(self, inputs):
        hidden = self.m(self.lnr1(self.d(inputs)))
        output = self.lnr2(self.d(hidden))
        return output
    
    def update_params(self, new_vals):
        for idx, param in enumerate(self.parameters()):
            param.data = new_vals[idx].clone()

class ClassifierNet(nn.Module):
    def __init__(self, hidden_size, output_size, resfile=None, ffwdfile=None, dropout=0.0, freeze=True, bias=True, pool=None):
        super(ClassifierNet, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.resfile = resfile
        self.ffwdfile = ffwdfile
        self.freeze = freeze
        self.dropout = dropout
        self.bias = bias
        self.pool = pool

        self.resnet = models.resnet18(pretrained=True)
        self.embed_size = self.resnet.fc.weight.shape[1]
        self.resnet.fc = nn.Sequential(nn.Dropout(self.dropout), nn.Linear(self.embed_size, self.output_size, bias=self.bias))

        if self.resfile is not None and os.path.exists(self.resfile):
            saved_state = torch.load(self.resfile, map_location=lambda storage, loc: storage)
            self.resnet.load_state_dict(saved_state)

        if self.freeze:
            for param in self.resnet.parameters():
                param.requires_grad = False

        if self.ffwdfile is not None:
            self.resnet.fc = nn.Identity()
            self.ff = FeedForwardNet(self.embed_size, self.hidden_size, self.output_size, dropout=self.dropout, bias=self.bias)
            if os.path.exists(self.ffwdfile):
                saved_state = torch.load(self.ffwdfile, map_location=lambda storage, loc: storage)
                self.ff.load_state_dict(saved_state)
        else:
            self.ff = None
        
    def forward(self, x):
        if self.pool is not None:
            batch_size = x.shape[0]
            num_tiles = x.shape[1]
            x = x.contiguous().view(-1, x.shape[-3], x.shape[-2], x.shape[-1])
            
        y = self.resnet(x)

        if self.ff is not None:
            y = self.ff(y)
            
        if self.pool is not None:
            y = y.contiguous().view(batch_size, num_tiles, -1)
            y = self.pool(y, dim=1)
        
        return y