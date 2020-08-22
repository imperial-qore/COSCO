import os
from .constants import *
import numpy as np
import torch
import random
import matplotlib.pyplot as plt

from sys import argv

plt.style.use(['science', 'ieee'])
plt.rcParams["text.usetex"] = False
plt.rcParams['figure.figsize'] = 2, 2

class color:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    RED = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def plot_accuracies(accuracy_list):
	value_loss = [i[1] for i in accuracy_list]
	policy_loss = [-1*i[2] for i in accuracy_list]
	loss = [i[1]-i[2] for i in accuracy_list]
	plt.xlabel('Epochs')
	plt.ylabel('Average Value Loss')
	plt.plot(range(len(value_loss)), value_loss, label='Average Value Loss', linewidth=1, linestyle='-', marker='.')
	plt.legend(loc=1)
	plt.savefig('value-graph.pdf')
	plt.clf()
	plt.xlabel('Epochs')
	plt.ylabel('Average Policy Loss')
	plt.errorbar(range(len(policy_loss)), policy_loss, label='Average Policy Loss', alpha = 0.7,\
	    linewidth = 1, linestyle='dotted', marker='+')
	plt.legend(loc=4)
	plt.savefig('policy-graph.pdf')
	plt.clf()
	plt.xlabel('Epochs')
	plt.ylabel('Average Loss')
	plt.errorbar(range(len(loss)), loss, label='Average Loss', alpha = 0.7,\
	    linewidth = 1, linestyle='dotted', marker='.')
	plt.legend(loc=4)
	plt.savefig('loss-graph.pdf')
	plt.clf()
