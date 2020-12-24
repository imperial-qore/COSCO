from .Scheduler import *
from .A2C.rl import *
import numpy as np
from copy import deepcopy
from math import sqrt
from random import randint

class PONDScheduler(Scheduler):
	def __init__(self, data_type):
		super().__init__()
		self.data_type = data_type
		self.num_hosts = int(data_type.split('_')[-1])
		self.r = {
					'shreshthtuli/yolo': [0]*self.num_hosts, 
					'shreshthtuli/pocketsphinx': [0]*self.num_hosts, 
					'shreshthtuli/aeneas':[0]*self.num_hosts
				}
		self.n = {
					'shreshthtuli/yolo': [0.001]*self.num_hosts, 
					'shreshthtuli/pocketsphinx': [0.001]*self.num_hosts, 
					'shreshthtuli/aeneas':[0.001]*self.num_hosts
				}
		self.apps = ['shreshthtuli/yolo', 'shreshthtuli/pocketsphinx', 'shreshthtuli/aeneas']
		self.default_app = self.apps[0]

	def calc_weight(self, alloc):
		weight = 0
		for cid, hid in alloc:
			# UCB weight calculation
			app = self.env.containerlist[cid].application if hasattr(self.env.containerlist[cid], 'application') else self.default_app
			weight += 10 * self.r[app][hid] * sqrt(np.log(100) / self.n[app][hid])
		return weight

	def get_last_value(self):
		if not self.env.stats: return 0
		energy = self.env.stats.metrics[-1]['energytotalinterval']
		all_energies = [e['energytotalinterval'] for e in self.env.stats.metrics[-20:]]
		all_latencies = [e['avgresponsetime'] for e in self.env.stats.metrics[-20:]]
		min_e, max_e = min(all_energies), max(all_energies)
		energy = (energy - min_e)/(0.1 + max_e - min_e)
		latency = self.env.stats.metrics[-1]['avgresponsetime']
		latency = (latency)/(0.1 + max(all_latencies))
		return Coeff_Energy*energy + Coeff_Latency*latency

	def update_r(self, alloc):
		new_r = -1 * self.get_last_value()
		for cid, hid in alloc:
			app = self.env.containerlist[cid].application if hasattr(self.env.containerlist[cid], 'application') else self.default_app
			self.r[app][hid] = self.r[app][hid] * self.n[app][hid] + new_r
		for cid, hid in alloc:
			app = self.env.containerlist[cid].application if hasattr(self.env.containerlist[cid], 'application') else self.default_app
			self.n[app][hid] += 1
		for cid, hid in alloc:
			app = self.env.containerlist[cid].application if hasattr(self.env.containerlist[cid], 'application') else self.default_app
			self.r[app][hid] = self.r[app][hid] / self.n[app][hid]

	def run_POND(self):
		# print(self.r)
		alloc = []; prev_alloc = {}
		for c in self.env.containerlist:
			if c: prev_alloc[c.id] = c.getHostID()
			if c and c.getHostID() != -1: hid = c.getHostID()
			else: hid = np.random.randint(0,self.num_hosts)
			if c: alloc.append((c.id, hid))
		cur_weight = self.calc_weight(alloc)
		# Max weight loop
		for i, j in enumerate(alloc):
			cid, hid = j
			new_alloc = deepcopy(alloc)
			for new_hid in range(0, self.num_hosts):
				if new_hid == hid: continue
				new_alloc[i] = (new_alloc[i][0], new_hid)
				new_weight = self.calc_weight(new_alloc)
				if new_weight > cur_weight and randint(1,100) < 5:
					cur_weight, alloc = new_weight, new_alloc
		decision = []
		for cid, hid in alloc:
			if prev_alloc[cid] != hid: decision.append((cid, hid))
		self.update_r(alloc)
		return decision

	def selection(self):
		return []

	def placement(self, containerIDs):
		decision = self.run_POND()
		return decision