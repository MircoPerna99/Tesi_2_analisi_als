import torch
import torch.nn as nn

class NeuralFactorizationMachineModel(torch.nn.Module):

    def __init__(self, n=None, k=None, dropout=0.2):
        super().__init__()
        # Initially we fill V with random values sampled from Gaussian distribution
        # NB: use nn.Parameter to compute gradients
        self.V = nn.Parameter(torch.randn(n, k),requires_grad=True)
        self.lin = nn.Linear(n, 1)
        self.fc1 = nn.Linear(k, 4*k)
        self.fc2 = nn.Linear(4*k, 2*k)
        self.fc3 = nn.Linear(2*k, k)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
        self.pl = nn.Linear(k, 1, bias=False)
        self.layer_norm0 = nn.LayerNorm(n)
        self.layer_norm1 = nn.LayerNorm(k)
        self.layer_norm2 = nn.LayerNorm(4*k)
        self.layer_norm3 = nn.LayerNorm(2*k)
        self.layer_norm4 = nn.LayerNorm(k)
        self.tanh = nn.Tanh()
        
    def forward(self, x):
            #Bi-Interactions
            #x = self.layer_norm0(x)
            out_1 = (x.unsqueeze(-1)*  self.V.unsqueeze(0)).sum(dim=1).pow(2) #S_1^2
            out_2 = ((x.unsqueeze(-1)).pow(2) * self.V.unsqueeze(0).pow(2)).sum(dim=1) # S_2
            out_inter = 0.5*(out_1 - out_2)
            out_inter = self.layer_norm1(out_inter)
            
            #Hidden Layer
            xc = self.fc1(out_inter)
            xc = self.layer_norm2(xc)
            xc = self.relu(xc)
            xc = self.dropout(xc)
            xc = self.fc2(xc)
            xc = self.layer_norm3(xc)
            xc = self.relu(xc)
            xc = self.dropout(xc)
            xc = self.fc3(xc)
            xc = self.layer_norm4(xc)
            xc = self.relu(xc)      
            xc = self.dropout(xc)
            #Prediction Layer
            out_n = self.pl(xc)
            out_lin = self.lin(x)
            out = out_n + out_lin
            
            return out
