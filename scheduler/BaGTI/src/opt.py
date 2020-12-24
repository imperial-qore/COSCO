import random 
import torch
import numpy as np
from copy import deepcopy
from src.constants import *
from src.optim_adahessian import Adahessian
import matplotlib.pyplot as plt

def convertToOneHot(dat, cpu_old, HOSTS):
    alloc = []
    for i in dat:
        oneHot = [0] * HOSTS; alist = i.tolist()[-HOSTS:]
        oneHot[alist.index(max(alist))] = 1; alloc.append(oneHot)
    new_dat_oneHot = torch.cat((cpu_old, torch.FloatTensor(alloc)), dim=1)
    return new_dat_oneHot

def opt(init, model, bounds, data_type):
    HOSTS = int(data_type.split('_')[-1])
    optimizer = torch.optim.AdamW([init] , lr=0.8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    iteration = 0; equal = 0; z_old = 100; zs = []
    while iteration < 200:
        cpu_old = deepcopy(init.data[:,0:-HOSTS]); alloc_old = deepcopy(init.data[:,-HOSTS:])
        z = model(init)
        optimizer.zero_grad(); z.backward(); optimizer.step(); scheduler.step()
        init.data = convertToOneHot(init.data, cpu_old, HOSTS)
        equal = equal + 1 if torch.all(alloc_old.eq(init.data[:,-HOSTS:])) else 0
        if equal > 30: break
        iteration += 1; z_old = z.item()
    #     zs.append(z.item())
    # plt.plot(zs); plt.show(); plt.clf()
    init.requires_grad = False 
    return init.data, iteration, model(init)

def so_opt(init, model, bounds, data_type):
    HOSTS = int(data_type.split('_')[-1])
    optimizer = Adahessian([init] , lr=0.8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    iteration = 0; equal = 0; z_old = 100; zs = []
    while iteration < 200:
        cpu_old = deepcopy(init.data[:,0:-HOSTS]); alloc_old = deepcopy(init.data[:,-HOSTS:])
        z = model(init)
        optimizer.zero_grad(); z.backward(create_graph=True); optimizer.step(); scheduler.step()
        init.data = convertToOneHot(init.data, cpu_old, HOSTS)
        equal = equal + 1 if torch.all(alloc_old.eq(init.data[:,-HOSTS:])) else 0
        if equal > 30: break
        iteration += 1; z_old = z.item()
    #     zs.append(z.item())
    # plt.plot(zs); plt.show(); plt.clf()
    init.requires_grad = False 
    return init.data, iteration, model(init)