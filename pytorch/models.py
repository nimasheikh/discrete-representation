from torch import nn as nn
import torch


class SimpMLP(nn.Module):

    def __init__(self, noise_var=0.5) -> None:
        super().__init__()
        self.rep_layers = nn.Sequential(
            nn.Linear(784, 500), 
            nn.ReLU(), 
            nn.Linear(500,100), 
            nn.Sigmoid()
        )
        self.classification_layer = nn.Linear(100, 10)
        self.noise_var = noise_var
        self.status='Train'
    def test(self):
        self.status="Test"
    def train(self):
        self.status="train"
    def forward(self, x):
        rep = self.rep_layers(x)
        if self.status=='Test':
            noise = self.noise_var * torch.rand(rep.shape)
        else:
            noise = torch.zeros(rep.shape)
        rep_noisy = rep + noise
        logits = self.classification_layer(rep_noisy)
        return logits
    
