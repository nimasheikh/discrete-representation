from torch import nn as nn
import torch


class SimpMLP(nn.Module):

    def __init__(self, noise_var=0.5) -> None:
        super().__init__()
        self.rep_layers = nn.Sequential(
            nn.Linear(784, 500), 
            nn.ReLU(), 
            nn.Linear(500,10), 
            nn.Sigmoid()
        )
        self.classification_layer = nn.Linear(10, 10)
        self.noise_var = noise_var
        self.status='Train'
    def test(self):
        self.status="Test"
    def train(self):
        self.status="Train"
    
    def representation_rounded(self, x):
        rep = self.rep_layers(x)
        rep_ = torch.round(rep)
        err = rep - rep_
        return rep_, err
    def representation(self, x):
        return self.rep_layers(x)
    
    def forward(self, x):
        rep = self.rep_layers(x)
        if self.status=='Train':
            noise = self.noise_var * torch.rand(rep.shape)
        elif self.status=='Test':
            noise = torch.zeros(rep.shape)

        rep_noisy = rep + noise
        logits = self.classification_layer(rep_noisy)
        return logits
    
