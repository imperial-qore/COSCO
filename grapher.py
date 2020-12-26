# 1. Response time with time for each application (scatter) --> 95th percentile for SOTA = SLA (scatter)
# 2. Number of migrations, Avrage interval response time, Average interval energy, scheduling time (time series)
# 3. Response time vs total IPS, Response time / Total IPS vs Total IPS (series)
# 4. Total energy, avg response time, cost/number of tasks completed, cost, number of total tasks completed
# Total number of migrations, total migration time, total execution, total scheduling time.

# Estimates of GOBI vs GOBI* (accuracy)

import matplotlib.pyplot as plt
import matplotlib
import itertools
import statistics
import pickle
import numpy as np
import scipy.stats
from stats.Stats import *
import seaborn as sns
from pprint import pprint
from utils.Utils import *
import os
import fnmatch
from sys import argv

plt.style.use(['science', 'ieee'])
plt.rcParams["text.usetex"] = True
size = (2.9, 2.5)
env = argv[1]
option = 0
sla_baseline = 'A3C'
if len(argv) >= 3:
	if 'SO' in argv[2]: option = 1
	elif 'H' in argv[2]: 
		option = 2
		sla_baseline = 'GOBI'

def fairness(l):
	a = 1 / (np.mean(l)-(scipy.stats.hmean(l)+0.001)) # 1 / slowdown i.e. 1 / (am - hm)
	if a: return a
	return 0

def jains_fairness(l):
	a = np.sum(l)**2 / (len(l) * np.sum(l**2)) # Jain's fairness index
	if a: return a
	return 0

def reduce(l):
	n = 5
	res = []
	for i in range(0, len(l)):
		res.append(statistics.mean(l[max(0, i-n):min(len(l), i+n)]))
	return res

def mean_confidence_interval(data, confidence=0.90):
    a = 1.0 * np.array(data)
    n = len(a)
    h = scipy.stats.sem(a) * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return h

PATH = 'all_datasets/' + env + '/'
SAVE_PATH = 'results/' + env + '/'

Models = ['GOBI*', 'GOBI', 'A3C', 'GA', 'POND', 'LR-MMT', 'MAD-MC'] 
if option == 1:
	Models = ['SOGOBI*', 'SOGOBI', 'GOBI*', 'GOBI', 'A3C', 'POND']
elif option == 2:
	Models = ['HGOBI*', 'HGOBI', 'GOBI*', 'GOBI', 'A3C', 'POND']
rot = 15
xLabel = 'Simulation Time (minutes)'
Colors = ['red', 'blue', 'green', 'orange', 'orchid', 'pink', 'cyan']
apps = ['yolo', 'pocketsphinx', 'aeneas']

yLabelsStatic = ['Total Energy (Kilowatt-hr)', 'Average Energy (Kilowatt-hr)', 'Interval Energy (Kilowatt-hr)', 'Average Interval Energy (Kilowatt-hr)',\
	'Number of completed tasks', 'Number of completed tasks per interval', 'Average Response Time (seconds)', 'Total Response Time (seconds)',\
	'Average Migration Time (seconds)', 'Total Migration Time (seconds)', 'Number of Task migrations', 'Average Wait Time (intervals)', 'Average Wait Time (intervals) per application',\
	'Average Completion Time (seconds)', 'Total Completion Time (seconds)', 'Average Response Time (seconds) per application',\
	'Cost per container (US Dollars)', 'Fraction of total SLA Violations', 'Fraction of SLA Violations per application', \
	'Interval Allocation Time (seconds)', 'Number of completed tasks per application', "Fairness (Jain's index)", 'Fairness', 'Fairness per application', \
	'Average CPU Utilization (%)', 'Average number of containers per Interval', 'Average RAM Utilization (%)', 'Scheduling Time (seconds)',\
	'Average Execution Time (seconds)']

yLabelStatic2 = {
	'Average Completion Time (seconds)': 'Number of completed tasks'
}

yLabelsTime = ['Interval Energy (Kilowatts)', 'Number of completed tasks', 'Interval Response Time (seconds)', \
	'Interval Migration Time (seconds)', 'Interval Completion Time (seconds)', 'Interval Cost (US Dollar)', \
	'Fraction of SLA Violations', 'Number of Task migrations', 'Number of Task migrations', 'Average Wait Time', 'Average Wait Time (intervals)', \
	'Average Execution Time (seconds)']

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

cost = (100 * 300 // 60) * (4 * 0.0472 + 2 * 0.189 + 2 * 0.166 + 2 * 0.333) # Hours * cost per hour

if env == 'framework':
	sla = {}
	r = all_stats[sla_baseline].allcontainerinfo[-1]
	start, end, application = np.array(r['start']), np.array(r['destroy']), np.array(r['application'])
	for app in apps:
		response_times = np.fmax(0, end - start)[application == 'shreshthtuli/'+app]
		response_times.sort()
		sla[app] = response_times[int(0.95*len(response_times))]
else:
	sla = {}
	r = all_stats[sla_baseline].allcontainerinfo[-1]
	start, end = np.array(r['start']), np.array(r['destroy'])
	response_times = np.fmax(0, end - start)
	response_times.sort()
	sla[apps[0]] = response_times[int(0.95*len(response_times))]
print(sla)

Data = dict()
CI = dict()

for ylabel in yLabelsStatic:
	Data[ylabel], CI[ylabel] = {}, {}
	for model in Models:
		# print(ylabel, model)
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
		if ylabel == 'Cost per container (US Dollars)':
			d = np.array([i['numdestroyed'] for i in stats.metrics]) if stats else np.array([0])
			Data[ylabel][model], CI[ylabel][model] = cost / float(np.sum(d)) if len(d) != 1 else 0, 0
		if 'f' in env and ylabel == 'Number of completed tasks per application':
			r = stats.allcontainerinfo[-1]['application'] if stats else []
			application = np.array(r)
			total = []
			for app in apps:
				total.append(len(application[application == 'shreshthtuli/'+app]))
			Data[ylabel][model], CI[ylabel][model] = total, [0]*3
		if ylabel == 'Number of completed tasks per interval':
			d = np.array([i['numdestroyed'] for i in stats.metrics]) if stats else np.array([0])
			Data[ylabel][model], CI[ylabel][model] = np.mean(d), mean_confidence_interval(d)
		if ylabel == 'Average Response Time (seconds)':
			d = np.array([max(0, i['avgresponsetime']) for i in stats.metrics]) if stats else np.array([0])
			d2 = np.array([i['numdestroyed'] for i in stats.metrics]) if stats else np.array([1])
			Data[ylabel][model], CI[ylabel][model] = np.mean(d[d2>0]), mean_confidence_interval(d[d2>0])
		if ylabel == 'Average Execution Time (seconds)':
			d = np.array([max(0, i['avgresponsetime']) for i in stats.metrics]) if stats else np.array([0])
			d1 = np.array([i['avgmigrationtime'] for i in stats.metrics]) if stats else np.array([0])
			d2 = np.array([i['numdestroyed'] for i in stats.metrics]) if stats else np.array([1])
			Data[ylabel][model], CI[ylabel][model] = np.mean(d[d2>0] - d1[d2>0]), mean_confidence_interval(d[d2>0] - d1[d2>0])
		if 'f' in env and ylabel == 'Average Response Time (seconds) per application':
			r = stats.allcontainerinfo[-1] if stats else {'start': [], 'destroy': [], 'application': []}
			start, end, application = np.array(r['start']), np.array(r['destroy']), np.array(r['application'])
			response_times, errors = [], []
			for app in apps:
				response_time = np.fmax(0, end[end!=-1] - start[end!=-1])[application[end!=-1] == 'shreshthtuli/'+app] *300
				response_times.append(np.mean(response_time))
				er = mean_confidence_interval(response_time)
				errors.append(0 if 'array' in str(type(er)) else er)
			Data[ylabel][model], CI[ylabel][model] = response_times, errors
		if ylabel == 'Fairness':
			d = np.array([fairness(np.array(i['ips'])) for i in stats.activecontainerinfo]) if stats else np.array([0])
			Data[ylabel][model], CI[ylabel][model] = np.mean(d), mean_confidence_interval(d)
		if ylabel == "Fairness (Jain's index)":
			d = np.array([jains_fairness(np.array(i['ips'])) for i in stats.activecontainerinfo]) if stats else np.array([0])
			Data[ylabel][model], CI[ylabel][model] = np.mean(d), mean_confidence_interval(d)
		if 'f' in env and ylabel == 'Fairness per application':
			r = stats.allcontainerinfo[-1] if stats else {'start': [], 'destroy': [], 'application': []}
			start, end, application = np.array(r['start']), np.array(r['destroy']), np.array(r['application'])
			response_times = []
			for app in apps:
				response_time = np.fmax(0, end[end!=-1] - start[end!=-1])[application[end!=-1] == 'shreshthtuli/'+app] *300
				er = 1/(np.mean(response_time)-scipy.stats.hmean(response_time))
				response_times.append(0 if 'array' in str(type(er)) else er)
			Data[ylabel][model], CI[ylabel][model] = response_times, [0]*3
		if ylabel == 'Total Response Time (seconds)':
			d = np.array([max(0, i['avgresponsetime']) for i in stats.metrics]) if stats else np.array([0.])
			d2 = np.array([i['numdestroyed'] for i in stats.metrics]) if stats else np.array([1])
			Data[ylabel][model], CI[ylabel][model] = np.sum(d[d2>0]*d2[d2>0]), 0
		if 'f' in env and ylabel == 'Fraction of total SLA Violations':
			r = stats.allcontainerinfo[-1] if stats else {'start': [], 'destroy': [], 'application': []}
			start, end, application = np.array(r['start']), np.array(r['destroy']), np.array(r['application'])
			violations, total = 0, 0
			for app in apps:
				response_times = np.fmax(0, end[end!=-1] - start[end!=-1])[application[end!=-1] == 'shreshthtuli/'+app]
				violations += len(response_times[response_times > sla[app]])
				total += len(response_times)
			Data[ylabel][model], CI[ylabel][model] = violations / (total+0.01), 0
		if 'f' not in env and ylabel == 'Fraction of total SLA Violations':
			r = stats.allcontainerinfo[-1] if stats else {'start': [], 'destroy': []}
			start, end = np.array(r['start']), np.array(r['destroy'])
			violations, total = 0, 0
			response_times = np.fmax(0, end[end!=-1] - start[end!=-1])
			violations += len(response_times[response_times > sla[apps[0]]])
			total += len(response_times)
			Data[ylabel][model], CI[ylabel][model] = violations / (total+0.01) if '*' not in model else 0, 0
		if 'f' in env and ylabel == 'Fraction of SLA Violations per application':
			r = stats.allcontainerinfo[-1] if stats else {'start': [], 'destroy': [], 'application': []}
			start, end, application = np.array(r['start']), np.array(r['destroy']), np.array(r['application'])
			violations = []
			for app in apps:
				response_times = np.fmax(0, end[end!=-1] - start[end!=-1])[application[end!=-1] == 'shreshthtuli/'+app]
				violations.append(len(response_times[response_times > sla[app]])/(len(response_times)+0.001))
			Data[ylabel][model], CI[ylabel][model] = violations, [0]*3
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
		if ylabel == 'Average Wait Time (intervals)':
			d = np.array([(np.average(i['waittime'])-1 if i != [] else 0) for i in stats.metrics]) if stats else np.array([0.])
			Data[ylabel][model], CI[ylabel][model] = np.sum(d[d>0]), mean_confidence_interval(d[d>0])
		if 'f' in env and ylabel == 'Average Wait Time (intervals)':
			r = stats.allcontainerinfo[-1] if stats else {'start': [], 'create': [], 'application': []}
			start, end, application = np.array(r['create']), np.array(r['start']), np.array(r['application'])
			response_times, errors = [], []
			response_time = np.fmax(0, end - start - 1)
			response_times = np.mean(response_time)
			er = mean_confidence_interval(response_time)
			errors = 0 if 'array' in str(type(er)) else er
			Data[ylabel][model], CI[ylabel][model] = response_times, errors
		if 'f' in env and ylabel == 'Average Wait Time (intervals) per application':
			r = stats.allcontainerinfo[-1] if stats else {'start': [], 'create': [], 'application': []}
			start, end, application = np.array(r['create']), np.array(r['start']), np.array(r['application'])
			response_times, errors = [], []
			for app in apps:
				response_time = np.fmax(0, end - start - 1)[application == 'shreshthtuli/'+app]
				response_times.append(np.mean(response_time))
				er = mean_confidence_interval(response_time)
				errors.append(0 if 'array' in str(type(er)) else er)
			Data[ylabel][model], CI[ylabel][model] = response_times, errors
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
pprint(Data)
# print(CI)

##### BAR PLOTS #####

for ylabel in yLabelsStatic:
	if Models[0] not in Data[ylabel]: continue
	if 'per application' in ylabel: continue
	print(color.BOLD+ylabel+color.ENDC)
	plt.figure(figsize=size)
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

for ylabel in yLabelsStatic:
	if Models[0] not in Data[ylabel]: continue
	if 'per application' not in ylabel: continue
	print(color.BOLD+ylabel+color.ENDC)
	plt.figure(figsize=size)
	plt.xlabel('Model')
	plt.ylabel(ylabel.replace('%', '\%'))
	if 'Wait' in ylabel: plt.gca().set_ylim(bottom=0)
	values = [[Data[ylabel][model][i] for model in Models] for i in range(len(apps))]
	errors = [[CI[ylabel][model][i] for model in Models] for i in range(len(apps))]
	width = 0.25
	x = np.arange(len(values[0]))
	for i in range(len(apps)):
		p1 = plt.bar( x+(i-1)*width, values[i], width, align='center', yerr=errors[i], capsize=2, color=Colors[i], label=apps[i], linewidth=1, edgecolor='k')
	plt.legend()
	plt.xticks(range(len(values[i])), Models, rotation=rot)
	plt.savefig(SAVE_PATH+'Bar-'+ylabel.replace(' ', '_')+".pdf")
	plt.clf()

# exit()

##### BOX PLOTS #####

Data = dict()
CI = dict()

for ylabel in yLabelsStatic:
	Data[ylabel], CI[ylabel] = {}, {}
	for model in Models:
		# print(ylabel, model)
		stats = all_stats[model]
		# Major metrics
		if ylabel == 'Average Energy (Kilowatt-hr)':
			d = np.array([i['energytotalinterval'] for i in stats.metrics])/1000 if stats else np.array([])
			d2 = np.array([i['numdestroyed'] for i in stats.metrics]) if stats else np.array([1])
			Data[ylabel][model], CI[ylabel][model] = d[d2>0]/d2[d2>0], 0
		if ylabel == 'Interval Energy (Kilowatt-hr)':
			d = np.array([i['energytotalinterval'] for i in stats.metrics])/1000 if stats else np.array([0])
			Data[ylabel][model], CI[ylabel][model] = d, mean_confidence_interval(d)
		if ylabel == 'Average Interval Energy (Kilowatt-hr)':
			d = np.array([i['energytotalinterval'] for i in stats.metrics])/1000 if stats else np.array([0])
			d2 = np.array([i['numdestroyed'] for i in stats.metrics]) if stats else np.array([1])
			Data[ylabel][model], CI[ylabel][model] = d[d2>0]/d2[d2>0], mean_confidence_interval(d[d2>0]/d2[d2>0])
		if ylabel == 'Number of completed tasks per interval':
			d = np.array([i['numdestroyed'] for i in stats.metrics]) if stats else np.array([0])
			Data[ylabel][model], CI[ylabel][model] = d, mean_confidence_interval(d)
		if ylabel == 'Average Response Time (seconds)':
			d = np.array([max(0, i['avgresponsetime']) for i in stats.metrics]) if stats else np.array([0])
			d2 = np.array([i['numdestroyed'] for i in stats.metrics]) if stats else np.array([1])
			Data[ylabel][model], CI[ylabel][model] = d[d2>0], mean_confidence_interval(d[d2>0])
		if ylabel == 'Average Execution Time (seconds)':
			d = np.array([max(0, i['avgresponsetime']) for i in stats.metrics]) if stats else np.array([0])
			d1 = np.array([i['avgmigrationtime'] for i in stats.metrics]) if stats else np.array([0])
			d2 = np.array([i['numdestroyed'] for i in stats.metrics]) if stats else np.array([1])
			Data[ylabel][model], CI[ylabel][model] = np.maximum(0, d[d2>0] - d1[d2>0]), mean_confidence_interval(d[d2>0] - d1[d2>0])
		if 'f' in env and ylabel == 'Average Response Time (seconds) per application':
			r = stats.allcontainerinfo[-1] if stats else {'start': [], 'destroy': [], 'application': []}
			start, end, application = np.array(r['start']), np.array(r['destroy']), np.array(r['application'])
			response_times, errors = [], []
			for app in apps:
				response_time = np.fmax(0, end[end!=-1] - start[end!=-1])[application[end!=-1] == 'shreshthtuli/'+app] *300
				response_times.append(response_time)
				er = mean_confidence_interval(response_time)
				errors.append(0 if 'array' in str(type(er)) else er)
			Data[ylabel][model], CI[ylabel][model] = response_times, errors
		# Auxilliary metrics
		if ylabel == 'Average Migration Time (seconds)':
			d = np.array([i['avgmigrationtime'] for i in stats.metrics]) if stats else np.array([0])
			d2 = np.array([i['numdestroyed'] for i in stats.metrics]) if stats else np.array([1])
			Data[ylabel][model], CI[ylabel][model] = d[d2>0], mean_confidence_interval(d[d2>0])
		if ylabel == 'Average Wait Time (intervals)':
			d = np.array([(np.average(i['waittime'])-1 if i != [] else 0) for i in stats.metrics]) if stats else np.array([0.])
			Data[ylabel][model], CI[ylabel][model] = d[d>0], mean_confidence_interval(d[d>0])
		if 'f' in env and ylabel == 'Average Wait Time (intervals)':
			r = stats.allcontainerinfo[-1] if stats else {'start': [], 'create': [], 'application': []}
			start, end, application = np.array(r['create']), np.array(r['start']), np.array(r['application'])
			response_times, errors = [], []
			response_time = np.fmax(0, end - start - 1)
			response_times = response_time
			er = mean_confidence_interval(response_time)
			errors = 0 if 'array' in str(type(er)) else er
			Data[ylabel][model], CI[ylabel][model] = response_times, errors
		if 'f' in env and ylabel == 'Average Wait Time (intervals) per application':
			r = stats.allcontainerinfo[-1] if stats else {'start': [], 'create': [], 'application': []}
			start, end, application = np.array(r['create']), np.array(r['start']), np.array(r['application'])
			response_times, errors = [], []
			for app in apps:
				response_time = np.fmax(0, end - start - 1)[application == 'shreshthtuli/'+app]
				response_times.append(response_time)
				er = mean_confidence_interval(response_time)
				errors.append(0 if 'array' in str(type(er)) else er)
			Data[ylabel][model], CI[ylabel][model] = response_times, errors
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


for ylabel in yLabelsStatic:
	if Models[0] not in Data[ylabel]: continue
	if 'per application' in ylabel: continue
	print(color.BLUE+ylabel+color.ENDC)
	plt.figure(figsize=size)
	plt.xlabel('Model')
	plt.ylabel(ylabel.replace('%', '\%'))
	values = [Data[ylabel][model] for model in Models]
	errors = [CI[ylabel][model] for model in Models]
	# plt.ylim(0, max(values)+statistics.stdev(values))
	p1 = plt.boxplot(values, positions=np.arange(len(values)), notch=False, showmeans=True, widths=0.65, meanprops=dict(marker='.', markeredgecolor='black', markerfacecolor='black'), showfliers=False)
	plt.xticks(range(len(values)), Models, rotation=rot)
	plt.savefig(SAVE_PATH+'Box-'+ylabel.replace(' ', '_')+".pdf")
	plt.clf()

for ylabel in yLabelsStatic:
	if Models[0] not in Data[ylabel]: continue
	if 'per application' not in ylabel: continue
	print(color.BLUE+ylabel+color.ENDC)
	plt.figure(figsize=size)
	plt.xlabel('Model')
	plt.ylabel(ylabel.replace('%', '\%'))
	if 'Wait' in ylabel: plt.gca().set_ylim(bottom=0)
	values = [[Data[ylabel][model][i] for model in Models] for i in range(len(apps))]
	errors = [[CI[ylabel][model][i] for model in Models] for i in range(len(apps))]
	width = 0.25
	x = np.arange(len(values[0]))
	for i in range(len(apps)):
		p1 = plt.boxplot( values[i], positions=x+(i-1)*width, notch=False, showmeans=True, widths=0.25, 
			meanprops=dict(marker='.', markeredgecolor='black', markerfacecolor='black'), showfliers=False)
		for param in ['boxes', 'whiskers', 'caps', 'medians']:
			plt.setp(p1[param], color=Colors[i])
		plt.plot([], '-', c=Colors[i], label=apps[i])
	plt.legend()
	plt.xticks(range(len(values[i])), Models, rotation=rot)
	plt.savefig(SAVE_PATH+'Box-'+ylabel.replace(' ', '_')+".pdf")
	plt.clf()

##### LINE PLOTS #####

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
			d = np.array([max(0, i['avgresponsetime']) for i in stats.metrics]) if stats else np.array([0])
			d2 = np.array([i['numdestroyed'] for i in stats.metrics]) if stats else np.array([1])
			Data[ylabel][model], CI[ylabel][model] = d/(d2+0.001), mean_confidence_interval(d/(d2+0.001))
		# SLA Violations, Cost (USD)
		# Auxilliary metrics
		if ylabel == 'Average Execution Time (seconds)':
			d = np.array([max(0, i['avgresponsetime']) for i in stats.metrics]) if stats else np.array([0])
			d1 = np.array([i['avgmigrationtime'] for i in stats.metrics]) if stats else np.array([0])
			d2 = np.array([i['numdestroyed'] for i in stats.metrics]) if stats else np.array([1])
			Data[ylabel][model], CI[ylabel][model] = np.array(d[d2>0] - d1[d2>0]), 0
		if ylabel == 'Average Migration Time (seconds)':
			d = np.array([i['avgmigrationtime'] for i in stats.metrics]) if stats else np.array([0])
			d2 = np.array([i['numdestroyed'] for i in stats.metrics]) if stats else np.array([1])
			Data[ylabel][model], CI[ylabel][model] = d/(d2+0.001), mean_confidence_interval(d/(d2+0.001))
		if ylabel == 'Average Wait Time (intervals)':
			d = np.array([(np.average(i['waittime'])-1 if i != [] else 0) for i in stats.metrics]) if stats else np.array([0.])
			d[np.isnan(d)] = 0
			Data[ylabel][model], CI[ylabel][model] = np.array(d), 0
		if ylabel == 'Number of Task migrations':
			d = np.array([i['nummigrations'] for i in stats.metrics]) if stats else np.array([0])
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
	plt.figure(figsize=size)
	plt.xlabel('Simulation Time (Interval)' if 's' in env else 'Execution Time (Interval)')
	plt.ylabel(ylabel.replace('%', '\%'))
	for model in Models:
		plt.plot(reduce(Data[ylabel][model]), color=Colors[Models.index(model)], linewidth=1.5, label=model, alpha=0.7)
	plt.legend()
	plt.savefig(SAVE_PATH+"Series-"+ylabel.replace(' ', '_')+".pdf")
	plt.clf()
