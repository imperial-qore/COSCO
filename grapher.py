# 1. Response time with time for each application (scatter) --> 95th percentile for SOTA = SLA (scatter)
# 2. Number of migrations, Avrage interval response time, Average interval energy, scheduling time (time series)
# 3. Response time vs total IPS, Response time / Total IPS vs Total IPS (series)
# 4. Total energy, avg response time, cost/number of tasks completed, cost, number of total tasks completed
# Total number of migrations, total migration time, total execution, total scheduling time.

# Estimates of GOBI vs GOBI* (accuracy)

import matplotlib.pyplot as plt
import itertools
import statistics
import pickle
import numpy as np
import scipy.stats
from stats.Stats import *
import seaborn as sns
from utils.Utils import *
import os
import fnmatch
from sys import argv

plt.style.use(['science', 'ieee'])
plt.rcParams["text.usetex"] = True

def reduce(l):
	n = 5
	res = []
	for i in range(0, len(l), n):
		res.append(statistics.mean(l[i:i+n]))
	return res

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    h = scipy.stats.sem(a) * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return h

env = argv[1]

PATH = 'all_datasets/' + env + '/'
SAVE_PATH = 'results/' + env + '/'

Models = ['GOBI*', 'GOBI', 'A3C', 'GA', 'LR-MMT', 'MAD-MC']
rot = 15
xLabel = 'Simulation Time (minutes)'
Colors = ['red', 'blue', 'green', 'orange', 'pink', 'cyan']

yLabelsStatic = ['Total Energy (Kilowatt-hr)', 'Average Energy (Kilowatt-hr)', 'Interval Energy (Kilowatt-hr)', 'Average Interval Energy (Kilowatt-hr)',\
	'Number of completed tasks', 'Number of completed tasks per interval', 'Average Response Time (seconds)', 'Total Response Time (seconds)',\
	'Average Migration Time (seconds)', 'Total Migration Time (seconds)', 'Number of Task migrations', 'Average Wait Time',\
	'Average Completion Time (seconds)', 'Total Completion Time (seconds)',\
	'Total Cost (US Dollar)', 'Fraction of SLA Violations', \
	'Interval Allocation Time (seconds)',\
	'Average CPU Utilization (%)', 'Average number of containers per Interval', 'Average RAM Utilization (%)', 'Scheduling Time (seconds)']

yLabelStatic2 = {
	'Average Completion Time (seconds)': 'Number of completed tasks'
}

yLabelsTime = ['Interval Energy (Kilowatts)', 'Number of completed tasks', 'Interval Response Time (seconds)', \
	'Interval Migration Time (seconds)', 'Interval Completion Time (seconds)', 'Interval Cost (US Dollar)', \
	'Fraction of SLA Violations', 'Number of Task migrations', 'Number of Task migrations']

all_stats_list = []
for model in Models:
	try:
		for file in os.listdir(PATH+model.replace('*', '2')):
			if fnmatch.fnmatch(file, '*.pk'):
				print(file)
				with open(PATH + model.replace('*', '2') + '/' + file, 'rb') as handle:
				    stats = pickle.load(handle)
				all_stats_list.append(stats)
				break
	except:
		all_stats_list.append(None)

all_stats = dict(zip(Models, all_stats_list))

Data = dict()
CI = dict()

for ylabel in yLabelsStatic:
	Data[ylabel], CI[ylabel] = {}, {}
	for model in Models:
		stats = all_stats[model]
		# Major metrics
		if ylabel == 'Total Energy (Kilowatt-hr)':
			d = np.array([i['energytotalinterval'] for i in stats.metrics])/1000 if stats else np.array([])
			Data[ylabel][model], CI[ylabel][model] = np.sum(d), 0
		if ylabel == 'Average Energy (Kilowatt-hr)':
			d = np.array([i['energytotalinterval'] for i in stats.metrics])/1000 if stats else np.array([])
			d2 = np.array([i['numdestroyed'] for i in stats.metrics]) if stats else np.array([1])
			Data[ylabel][model], CI[ylabel][model] = np.sum(d)/np.sum(d2), 0
		if ylabel == 'Interval Energy (Kilowatt-hr)':
			d = np.array([i['energytotalinterval'] for i in stats.metrics])/1000 if stats else np.array([0])
			Data[ylabel][model], CI[ylabel][model] = np.mean(d), mean_confidence_interval(d)
		if ylabel == 'Average Interval Energy (Kilowatt-hr)':
			d = np.array([i['energytotalinterval'] for i in stats.metrics])/1000 if stats else np.array([0])
			d2 = np.array([i['numdestroyed'] for i in stats.metrics]) if stats else np.array([1])
			Data[ylabel][model], CI[ylabel][model] = np.mean(d[d2>0]/d2[d2>0]), mean_confidence_interval(d[d2>0]/d2[d2>0])
		if ylabel == 'Number of completed tasks':
			d = np.array([i['numdestroyed'] for i in stats.metrics]) if stats else np.array([0])
			Data[ylabel][model], CI[ylabel][model] = np.sum(d), 0
		if ylabel == 'Number of completed tasks per interval':
			d = np.array([i['numdestroyed'] for i in stats.metrics]) if stats else np.array([0])
			Data[ylabel][model], CI[ylabel][model] = np.mean(d), mean_confidence_interval(d)
		if ylabel == 'Average Response Time (seconds)':
			d = np.array([i['avgresponsetime'] for i in stats.metrics]) if stats else np.array([0])
			d2 = np.array([i['numdestroyed'] for i in stats.metrics]) if stats else np.array([1])
			Data[ylabel][model], CI[ylabel][model] = np.mean(d[d2>0]), mean_confidence_interval(d[d2>0])
		if ylabel == 'Total Response Time (seconds)':
			d = np.array([i['avgresponsetime'] for i in stats.metrics]) if stats else np.array([0.])
			d2 = np.array([i['numdestroyed'] for i in stats.metrics]) if stats else np.array([1])
			Data[ylabel][model], CI[ylabel][model] = np.sum(d[d2>0]*d2[d2>0]), 0
		# SLA Violations, Cost (USD)
		# Auxilliary metrics
		if ylabel == 'Average Migration Time (seconds)':
			d = np.array([i['avgmigrationtime'] for i in stats.metrics]) if stats else np.array([0])
			d2 = np.array([i['numdestroyed'] for i in stats.metrics]) if stats else np.array([1])
			Data[ylabel][model], CI[ylabel][model] = np.mean(d[d2>0]), mean_confidence_interval(d[d2>0])
		if ylabel == 'Total Migration Time (seconds)':
			d = np.array([i['avgmigrationtime'] for i in stats.metrics]) if stats else np.array([0.])
			d2 = np.array([i['nummigrations'] for i in stats.metrics]) if stats else np.array([1])
			Data[ylabel][model], CI[ylabel][model] = np.sum(d[d2>0]*d2[d2>0]), 0
		if ylabel == 'Number of Task migrations':
			d = np.array([i['nummigrations'] for i in stats.metrics]) if stats else np.array([0])
			Data[ylabel][model], CI[ylabel][model] = np.sum(d), mean_confidence_interval(d)
		if ylabel == 'Average Wait Time':
			d = np.array([(np.average(i['waittime']) if i != [] else 0) for i in stats.metrics]) if stats else np.array([0.])
			Data[ylabel][model], CI[ylabel][model] = np.sum(d[d>0]), mean_confidence_interval(d[d>0])
		# Host metrics
		if ylabel == 'Average CPU Utilization (%)':
			d = np.array([(np.average(i['cpu']) if i != [] else 0) for i in stats.hostinfo]) if stats else np.array([0.])
			Data[ylabel][model], CI[ylabel][model] = np.sum(d), mean_confidence_interval(d)
		if ylabel == 'Average number of containers per Interval':
			d = np.array([(np.average(i['numcontainers']) if i != [] else 0.) for i in stats.hostinfo]) if stats else np.array([0.])
			Data[ylabel][model], CI[ylabel][model] = np.sum(d), mean_confidence_interval(d)
		if ylabel == 'Average RAM Utilization (%)':
			d = np.array([(np.average(100*np.array(i['ram'])/(np.array(i['ram'])+np.array(i['ramavailable']))) if i != [] else 0) for i in stats.hostinfo]) if stats else np.array([0.])
			Data[ylabel][model], CI[ylabel][model] = np.sum(d), mean_confidence_interval(d)
		# Scheduler metrics
		if ylabel == 'Scheduling Time (seconds)':
			d = np.array([i['schedulingtime'] for i in stats.schedulerinfo]) if stats else np.array([0.])
			Data[ylabel][model], CI[ylabel][model] = np.sum(d), mean_confidence_interval(d)
		if 'f' in env and ylabel == 'Interval Allocation Time (seconds)':
			d = np.array([i['migrationTime'] for i in stats.schedulerinfo]) if stats else np.array([0.])
			Data[ylabel][model], CI[ylabel][model] = np.sum(d), mean_confidence_interval(d)

# Bar Graphs
x = range(5,100*5,5)
print(Data)

for ylabel in yLabelsStatic:
	if Models[0] not in Data[ylabel]: continue
	print(color.BOLD+ylabel+color.ENDC)
	plt.xlabel('Model')
	plt.ylabel(ylabel.replace('%', '\%'))
	values = [Data[ylabel][model] for model in Models]
	errors = [CI[ylabel][model] for model in Models]
	plt.ylim(0, max(values)+statistics.stdev(values))
	p1 = plt.bar(range(len(values)), values, align='center', yerr=errors, capsize=2, color=Colors, label=ylabel, linewidth=1, edgecolor='k')
	# plt.legend()
	plt.xticks(range(len(values)), Models, rotation=rot)
	if ylabel in yLabelStatic2:
		plt.twinx()
		ylabel2 = yLabelStatic2[ylabel]
		plt.ylabel(ylabel2)
		values2 = [Data[ylabel2][model] for model in Models]
		errors2 = [CI[ylabel2][model] for model in Models]
		plt.ylim(0, max(values2)+10*statistics.stdev(values2))
		p2 = plt.errorbar(range(len(values2)), values2, color='black', alpha=0.7, yerr=errors2, capsize=2, label=ylabel2, marker='.', linewidth=2)
		plt.legend((p2[0],), (ylabel2,), loc=1)
	plt.savefig(SAVE_PATH+'Bar-'+ylabel.replace(' ', '_')+".pdf")
	plt.clf()

Data = dict()
CI = dict()

for ylabel in yLabelsStatic:
	Data[ylabel], CI[ylabel] = {}, {}
	for model in Models:
		stats = all_stats[model]
		# Major metrics
		if ylabel == 'Interval Energy (Kilowatt-hr)':
			d = np.array([i['energytotalinterval'] for i in stats.metrics])/1000 if stats else np.array([0])
			Data[ylabel][model], CI[ylabel][model] = d, mean_confidence_interval(d)
		if ylabel == 'Average Interval Energy (Kilowatt-hr)':
			d = np.array([i['energytotalinterval'] for i in stats.metrics])/1000 if stats else np.array([0])
			d2 = np.array([i['numdestroyed'] for i in stats.metrics]) if stats else np.array([1])
			Data[ylabel][model], CI[ylabel][model] = d/(d2+0.001), mean_confidence_interval(d/(d2+0.001))
		if ylabel == 'Number of completed tasks':
			d = np.array([i['numdestroyed'] for i in stats.metrics]) if stats else np.array([0])
			Data[ylabel][model], CI[ylabel][model] = d, 0
		if ylabel == 'Average Response Time (seconds)':
			d = np.array([i['avgresponsetime'] for i in stats.metrics]) if stats else np.array([0])
			d2 = np.array([i['numdestroyed'] for i in stats.metrics]) if stats else np.array([1])
			Data[ylabel][model], CI[ylabel][model] = d/(d2+0.001), mean_confidence_interval(d/(d2+0.001))
		# SLA Violations, Cost (USD)
		# Auxilliary metrics
		if ylabel == 'Average Migration Time (seconds)':
			d = np.array([i['avgmigrationtime'] for i in stats.metrics]) if stats else np.array([0])
			d2 = np.array([i['numdestroyed'] for i in stats.metrics]) if stats else np.array([1])
			Data[ylabel][model], CI[ylabel][model] = d/(d2+0.001), mean_confidence_interval(d/(d2+0.001))
		if ylabel == 'Number of Task migrations':
			d = np.array([i['nummigrations'] for i in stats.metrics]) if stats else np.array([0])
			Data[ylabel][model], CI[ylabel][model] = d, mean_confidence_interval(d)
		if ylabel == 'Average Wait Time':
			d = np.array([(np.average(i['waittime']) if i != [] else 0) for i in stats.metrics]) if stats else np.array([0.])
			Data[ylabel][model], CI[ylabel][model] = d, mean_confidence_interval(d)
		# Host metrics
		if ylabel == 'Average CPU Utilization (%)':
			d = np.array([(np.average(i['cpu']) if i != [] else 0) for i in stats.hostinfo]) if stats else np.array([0.])
			Data[ylabel][model], CI[ylabel][model] = d, mean_confidence_interval(d)
		if ylabel == 'Average number of containers per Interval':
			d = np.array([(np.average(i['numcontainers']) if i != [] else 0.) for i in stats.hostinfo]) if stats else np.array([0.])
			Data[ylabel][model], CI[ylabel][model] = d, mean_confidence_interval(d)
		if ylabel == 'Average RAM Utilization (%)':
			d = np.array([(np.average(100*np.array(i['ram'])/(np.array(i['ram'])+np.array(i['ramavailable']))) if i != [] else 0) for i in stats.hostinfo]) if stats else np.array([0.])
			Data[ylabel][model], CI[ylabel][model] = d, mean_confidence_interval(d)
		# Scheduler metrics
		if ylabel == 'Scheduling Time (seconds)':
			d = np.array([i['schedulingtime'] for i in stats.schedulerinfo]) if stats else np.array([0.])
			Data[ylabel][model], CI[ylabel][model] = d, mean_confidence_interval(d)
		if 'f' in env and ylabel == 'Interval Allocation Time (seconds)':
			d = np.array([i['migrationTime'] for i in stats.schedulerinfo]) if stats else np.array([0.])
			Data[ylabel][model], CI[ylabel][model] = d, mean_confidence_interval(d)


# Time series data
for ylabel in yLabelsStatic:
	if Models[0] not in Data[ylabel]: continue
	print(color.GREEN+ylabel+color.ENDC)
	plt.xlabel('Simulation Time (Hours)')
	plt.ylabel(ylabel.replace('%', '\%'))
	for model in Models:
		plt.plot(reduce(Data[ylabel][model]), color=Colors[Models.index(model)], linewidth=1, label=model, alpha=0.7)
	plt.legend()
	plt.savefig(SAVE_PATH+"Series-"+ylabel.replace(' ', '_')+".pdf")
	plt.clf()
