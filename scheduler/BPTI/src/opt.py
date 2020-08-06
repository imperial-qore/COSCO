import random 
import torch
import numpy as np
from copy import deepcopy
from src.constants import *

def opt(dataset, model, bounds, data_type):
    init = torch.tensor(random.choice(dataset)[0], dtype=torch.float, requires_grad=True)
    optimizer = torch.optim.AdamW([init] , lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    init.data[-1] = 0; iteration = 0; equal = 0; z_old = 100
    while True:
        if "a" not in data_type: 
            cpu_old = deepcopy(init.data[:,0]); alloc_old = deepcopy(init.data[:,1:])
        res = model(init)
        z = CoeffCd * res[1] - CoeffCl * res[0] if "a" in data_type else res
        optimizer.zero_grad()
        z.backward()
        optimizer.step()
        scheduler.step()
        # print(z.item())
        # if iteration % 100 == 0: print(init.data, res.data, z.data, init.grad.data)
        if "a" in data_type :
            for i in range(4):
                init.data[i] = max(bounds[i][0], min(init.data[i], bounds[i][1]))
        else:
            alloc = []
            for i in init.data:
                oneHot = [0] * 50; alist = i.tolist()[1:]
                oneHot[alist.index(max(alist))] = 1; alloc.append(oneHot)
            init.data = torch.cat((cpu_old.reshape(-1,1), torch.FloatTensor(alloc)), dim=1)
            equal = equal + 1 if torch.all(alloc_old.eq(init.data[:,1:])) else 0
        if ("a" in data_type and res[1] <= 0) or ("a" not in data_type and equal > 20): break
        if data_type not in ['airfoil', 'scheduler'] and z_old - z.item() < 1e-11: break
        iteration += 1; z_old = z.item()
    print("Iteration: {}\nResult: {}\nFitness: {}".format(iteration, 
          init.data, 
          z.data)) 