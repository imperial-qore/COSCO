import random 
import numpy as np
from copy import deepcopy
from .hgp_constants import *
from scipy import optimize
import warnings
warnings.filterwarnings("ignore")

gmodel = None

def convertToOneHot(dat, cpu_old, HOSTS):
    alloc = []
    for i in dat:
        oneHot = [0] * HOSTS; alist = i.tolist()[-HOSTS:]
        oneHot[alist.index(max(alist))] = 1; alloc.append(oneHot)
    new_dat_oneHot = np.concatenate((cpu_old, np.array(alloc)), axis=1)
    return new_dat_oneHot

def f(inp):
    x, s = gmodel.predict(inp.reshape(1,-1), return_std=True)
    return (x + UCB_K * s)[0]

def HGPopt(init, model, data_type):
    global gmodel
    gmodel = model
    HOSTS = int(data_type.split('_')[-1])
    init = init.reshape(HOSTS, HOSTS+2)
    cpu_old = deepcopy(init[:,0:-HOSTS]); alloc_old = deepcopy(init[:,-HOSTS:])
    init = optimize.minimize(f, x0=init.reshape(-1), tol=100, method='BFGS', options={'maxiter':1}).x
    init = init.reshape(HOSTS, HOSTS+2)
    init = convertToOneHot(init, cpu_old, HOSTS)
    return init, f(init)
