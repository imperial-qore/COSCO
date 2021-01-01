import matplotlib.pyplot as plt
import os
import pandas as pd 
import numpy as np
import torch
import random
import statistics

from sys import argv

class color:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    RED = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def load_energy_latency_data(HOSTS):
	dataset_path = '../BaGTI/datasets/energy_latency_'+str(HOSTS)+'_scheduling.csv'
	data = pd.read_csv(dataset_path) if os.path.exists(dataset_path) else pd.read_csv('scheduler/BaGTI/'+dataset_path)
	data = data.values.astype(np.float)
	max_ips_container = max(data.max(0)[HOSTS:2*HOSTS])
	dataset = []
	print("Dataset size", data.shape[0])
	for i in range(data.shape[0]):
		cpuH, cpuC, alloc = [], [], []
		for j in range(HOSTS):
			cpuH.append(data[i][j]/100)
			cpuC.append(data[i][j+HOSTS]/max_ips_container)
			oneHot = [0] * HOSTS
			if int(data[i][j+(2*HOSTS)]) >= 0: oneHot[int(data[i][j+(2*HOSTS)])] = 1
			alloc.append(oneHot)
		cpuH = np.array([cpuH]).transpose(); cpuC = np.array([cpuC]).transpose()
		alloc = np.array(alloc)
		dataset.append(((np.concatenate((cpuH, cpuC, alloc), axis=1)), np.array([(data[i][-2]- data.min(0)[-2])/(data.max(0)[-2] - data.min(0)[-2]), max(0, data[i][-1])/data.max(0)[-1]])))
		if dataset[-1][1][1] > 1:
			print(dataset[-1])
		# Normalization by (x - min)/(max - min)
	return dataset, len(dataset), max_ips_container
