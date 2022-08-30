import matplotlib.pyplot as plt
import os
from src.constants import *
import pandas as pd 
import numpy as np
import torch
import random
import statistics

from sys import argv

plt.style.use(['science', 'ieee'])
plt.rcParams["text.usetex"] = True
plt.rcParams['figure.figsize'] = 2, 1.2

if 'train' in argv[0] and not os.path.exists(MODEL_SAVE_PATH):
	os.mkdir(MODEL_SAVE_PATH)

def reduce(l):
	n = 10
	res = []
	low, high = [], []
	for i in range(0, len(l)):
		res.append(statistics.mean(l[max(0, i-n):min(len(l), i+n)]))
		low.append(min(l[max(0, i-n):min(len(l), i+n)])); high.append(max(l[max(0, i-n):min(len(l), i+n)]))
	return res, low, high

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
	data = pd.read_csv(dataset_path) if os.path.exists(dataset_path) else pd.read_csv('scheduler/BaGTI/'+dataset_path)
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

def load_energy_latency_data(HOSTS):
	dataset_path = 'datasets/energy_latency_'+str(HOSTS)+'_scheduling.csv'
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
		dataset.append(((np.concatenate((cpuH, cpuC, alloc), axis=1)), torch.Tensor([(data[i][-2]- data.min(0)[-2])/(data.max(0)[-2] - data.min(0)[-2]), max(0, data[i][-1])/data.max(0)[-1]])))
		if dataset[-1][1][1] > 1:
			print(dataset[-1])
		# Normalization by (x - min)/(max - min)
	return dataset, len(dataset), max_ips_container

def load_energy_latency2_data(HOSTS):
	dataset_path = 'datasets/energy_latency2_'+str(HOSTS)+'_scheduling.csv'
	data = pd.read_csv(dataset_path, header=None) if os.path.exists(dataset_path) else pd.read_csv('scheduler/BaGTI/'+dataset_path, header=None)
	data = data.values.astype(np.float)
	max_ips_container = max(data.max(0)[HOSTS:2*HOSTS])
	max_energy = data.max(0)[3*HOSTS]
	max_response = data.max(0)[3*HOSTS+1]
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
		pred_vals = np.broadcast_to(np.array([data[i][3*HOSTS]/max_energy, data[i][3*HOSTS+1]/max_response]), (HOSTS, 2))
		dataset.append(((np.concatenate((cpuH, cpuC, alloc, pred_vals), axis=1)), torch.Tensor([(data[i][-2]- data.min(0)[-2])/(data.max(0)[-2] - data.min(0)[-2]), max(0, data[i][-1])/data.max(0)[-1]])))
		# Normalization by (x - min)/(max - min)
	return dataset, len(dataset), (max_ips_container, max_energy, max_response)

def load_stochastic_energy_latency_data(HOSTS):
	return load_energy_latency_data(HOSTS)

def load_stochastic_energy_latency2_data(HOSTS):
	return load_energy_latency2_data(HOSTS)

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
	plt.clf()
	plt.xlabel('Epochs')
	plt.ylabel('Testing Loss')
	a, b, c = reduce(testAcc)
	b2, _, _ = reduce(b); c2, _, _ = reduce(c)
	plt.fill_between(np.arange(len(testAcc)), b2, c2, color='lightgreen', alpha=.5)
	plt.plot(a, label='Testing Loss', alpha = 0.7, color='g',\
	    linewidth = 1, linestyle='-')
	plt.savefig('graphs/'+data_type+'/reduced.pdf')
