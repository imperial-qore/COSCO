import random 
import torch
import numpy as np
from copy import deepcopy
from src.constants import *
from src.adahessian import Adahessian
from src.opt import *
import matplotlib.pyplot as plt
import dgl

def optW(init, apps, graph, model, bounds, data_type):
    HOSTS = int(data_type.split('_')[-1])
    optimizer = torch.optim.AdamW([init] , lr=0.8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    iteration = 0; equal = 0; z_old = 100; zs = []
    while iteration < 200:
        cpu_old = deepcopy(init.data[:,0:-HOSTS]); alloc_old = deepcopy(init.data[:,-HOSTS:])
        z = model(init, apps, graph)
        optimizer.zero_grad(); z.backward(); optimizer.step(); scheduler.step()
        init.data = convertToOneHot(init.data, cpu_old, HOSTS)
        equal = equal + 1 if torch.all(alloc_old.eq(init.data[:,-HOSTS:])) else 0
        if equal > 30: break
        iteration += 1; z_old = z.item()
    #     zs.append(z.item())
    # plt.plot(zs); plt.show(); plt.clf()
    init.requires_grad = False 
    return init.data, iteration, model(init, apps, graph)
