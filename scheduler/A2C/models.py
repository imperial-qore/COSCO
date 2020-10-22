import torch
import torch.nn as nn
from .constants import *

from sys import argv

class energy_50_RL(nn.Module):
    def __init__(self):
        super(energy_50_RL, self).__init__()
        self.name = "energy_RL"
        self.feature = nn.Sequential(
            nn.Linear(50 * 51, 128),
            nn.Softplus(),
            nn.Linear(128, 128),
            nn.Softplus())
        self.value = nn.Sequential(
            nn.Linear(128, 64), 
            nn.Tanhshrink(),
            nn.Linear(64, 1),
            nn.Sigmoid())
        self.action = nn.Sequential(
            nn.Linear(128, 256), 
            nn.Softplus(),
            nn.Linear(256, 50 * 50))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.flatten()
        x = self.feature(x)
        value = self.value(x)
        action = self.softmax(self.action(x).reshape(50,50))
        return value, action

class energy_latency_10_RL(nn.Module):
    def __init__(self):
        super(energy_latency_10_RL, self).__init__()
        self.name = "energy_latency_10_"+str(Coeff_Energy)+"_"+str(Coeff_Latency)+"_RL"
        self.feature = nn.Sequential(
            nn.Linear(10 * 12, 128),
            nn.Softplus(),
            nn.Linear(128, 128),
            nn.Softplus())
        self.value = nn.Sequential(
            nn.Linear(128, 64), 
            nn.Tanhshrink(),
            nn.Linear(64, 1),
            nn.Sigmoid())
        self.action = nn.Sequential(
            nn.Linear(128, 256), 
            nn.Softplus(),
            nn.Linear(256, 10 * 10))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.flatten()
        x = self.feature(x)
        value = self.value(x)
        action = self.softmax(self.action(x).reshape(10,10))
        return value, action

class energy_latency_50_RL(nn.Module):
    def __init__(self):
        super(energy_latency_50_RL, self).__init__()
        self.name = "energy_latency_50_"+str(Coeff_Energy)+"_"+str(Coeff_Latency)+"_RL"
        self.feature = nn.Sequential(
            nn.Linear(50 * 52, 128),
            nn.Softplus(),
            nn.Linear(128, 128),
            nn.Softplus())
        self.value = nn.Sequential(
            nn.Linear(128, 64), 
            nn.Tanhshrink(),
            nn.Linear(64, 1),
            nn.Sigmoid())
        self.action = nn.Sequential(
            nn.Linear(128, 256), 
            nn.Softplus(),
            nn.Linear(256, 10 * 10))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.flatten()
        x = self.feature(x)
        value = self.value(x)
        action = self.softmax(self.action(x).reshape(10,10))
        return value, action

