import torch
import torch.nn as nn

class FeedForwardNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, initial_vals=None, dropout=0.0, bias=True):
        super(Net, self).__init__()
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