import torch
import torch.nn as nn
from torchvision import models

class FeedForwardNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, initial_vals=None, dropout=0.0, bias=True):
        super(FeedForwardNet, self).__init__()
        self.d = nn.Dropout(dropout)
        self.m = nn.Tanh()
        self.lnr1 = nn.Linear(input_size, hidden_size, bias=bias)
        self.lnr2 = nn.Linear(hidden_size, output_size, bias=bias)
        
        if initial_vals is not None:
            self.update_params(initial_vals)
        
    def forward(self, inputs):
        hidden = self.m(self.lnr1(self.d(inputs)))
        output = self.lnr2(self.d(hidden))
        return output
    
    def update_params(self, new_vals):
        for idx, param in enumerate(self.parameters()):
            param.data = new_vals[idx].clone()

class ClassifierNet(nn.Module):
    def __init__(self, new_hidden_size, output_size, resfile=None, ffwdfile=None, freeze=True, dropout=0.0):
        super(ClassifierNet, self).__init__()
        self.new_hidden_size = new_hidden_size
        self.output_size = output_size
        self.resfile = resfile
        self.ffwdfile = ffwdfile
        self.freeze = freeze
        self.dropout = dropout

        self.resnet = models.resnet18(pretrained=True)
        self.hidden_size = self.resnet.fc.weight.shape[1]
        self.resnet.fc = nn.Linear(self.hidden_size, self.output_size, bias=True)

        if self.resfile is not None:
            saved_state = torch.load(self.resfile, map_location=lambda storage, loc: storage)
            self.resnet.load_state_dict(saved_state)
        
        self.resnet.fc = nn.Identity()

        if self.freeze:
            for param in self.resnet.parameters():
                param.requires_grad = False

        self.ff = FeedForwardNet(self.hidden_size, self.new_hidden_size, self.output_size, dropout=self.dropout)

        if self.ffwdfile is not None:
            saved_state = torch.load(self.ffwdfile, map_location=lambda storage, loc: storage)
            self.ff.load_state_dict(saved_state)
        
    def forward(self, x):
        y = self.resnet(x)
        z = self.ff(y)
        
        return z