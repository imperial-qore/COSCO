import matplotlib.pyplot as plt
import os
from src.constants import *
import pandas as pd 
import numpy as np
import torch
import random

from sys import argv

plt.style.use(['science', 'ieee'])
plt.rcParams["text.usetex"] = True
plt.rcParams['figure.figsize'] = 2, 2

if 'train' in argv[0] and not os.path.exists(MODEL_SAVE_PATH):
	os.mkdir(MODEL_SAVE_PATH)

class color:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    RED = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def load_energy_data():
	dataset_path = 'datasets/energy_scheduling.csv'
	data = pd.read_csv(dataset_path) if os.path.exists(dataset_path) else pd.read_csv('scheduler/BPTI/'+dataset_path)
	data = data.values.astype(np.float)
	dataset = []
	print("Dataset size", data.shape[0])
	for i in range(data.shape[0]):
		cpu, alloc = [], []
		for j in range(50):
			cpu.append(data[i][j]/100)
			oneHot = [0] * 50
			if int(data[i][j+50]) >= 0: oneHot[int(data[i][j+50])] = 1
			alloc.append(oneHot)
		cpu = np.array([cpu]).transpose()
		alloc = np.array(alloc)
		dataset.append(((np.concatenate((cpu, alloc), axis=1)), torch.Tensor([(data[i][-1]- 9800)/9000])))
		# Normalization by (x - min)/(max - min)
	return dataset, len(dataset)

def load_energy_latency_data():
	dataset_path = 'datasets/energy_latency_scheduling.csv'
	data = pd.read_csv(dataset_path) if os.path.exists(dataset_path) else pd.read_csv('scheduler/BPTI/'+dataset_path)
	data = data.values.astype(np.float)
	dataset = []
	print("Dataset size", data.shape[0])
	for i in range(data.shape[0]):
		cpuH, cpuC, alloc = [], [], []
		for j in range(50):
			cpuH.append(data[i][j]/100)
			cpuC.append(data[i][j+50]/6000)
			oneHot = [0] * 50
			if int(data[i][j+100]) >= 0: oneHot[int(data[i][j+100])] = 1
			alloc.append(oneHot)
		cpuH = np.array([cpuH]).transpose(); cpuC = np.array([cpuC]).transpose()
		alloc = np.array(alloc)
		dataset.append(((np.concatenate((cpuH, cpuC, alloc), axis=1)), torch.Tensor([(data[i][-2]- 9800)/9000, (data[i][-1])/7000])))
		# Normalization by (x - min)/(max - min)
	return dataset, len(dataset)

def plot_accuracies(accuracy_list, data_type):
	trainAcc = [i[0] for i in accuracy_list]
	testAcc = [i[1] for i in accuracy_list]
	plt.xlabel('Epochs')
	plt.ylabel('Average Training Loss')
	plt.plot(range(len(trainAcc)), trainAcc, label='Average Training Loss', linewidth=1, linestyle='-', marker='.')
	plt.legend(loc=1)
	plt.savefig('graphs/'+data_type+'/training-graph.pdf')
	plt.clf()
	plt.xlabel('Epochs')
	plt.ylabel('Average Testing Loss')
	plt.errorbar(range(len(testAcc)), testAcc, label='Average Testing Loss', alpha = 0.7,\
	    linewidth = 1, linestyle='dotted', marker='+')
	plt.legend(loc=4)
	plt.savefig('graphs/'+data_type+'/testing-graph.pdf')
