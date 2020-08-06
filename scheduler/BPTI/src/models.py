import torch
import torch.nn as nn

class energy(nn.Module):
    def __init__(self):
        super(energy, self).__init__()
        self.name = "energy"
        self.find = nn.Sequential(
            nn.Linear(50 * 51, 128),
            nn.Softplus(),
            nn.Linear(128, 128),
            nn.Softplus(),
            nn.Linear(128, 64), 
            nn.Tanhshrink(),
            nn.Linear(64, 1),
            nn.Sigmoid())

    def forward(self, x):
        x = x.flatten()
        x = self.find(x)
        return x
