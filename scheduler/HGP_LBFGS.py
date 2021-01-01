import sys
sys.path.append('scheduler/HGP/')

from .Scheduler import *
from .HGP.train import *

class HGPScheduler(Scheduler):
	def __init__(self, data_type):
		super().__init__()
		self.model, _, _, self.max_container_ips = load_model(data_type)
		self.data_type = data_type
		self.hosts = int(data_type.split('_')[-1])

	def run_HGP(self):
		cpu = [host.getCPU()/100 for host in self.env.hostlist]
		cpu = np.array([cpu]).transpose()
		cpuC = [(c.getApparentIPS()/self.max_container_ips if c else 0) for c in self.env.containerlist]
		cpuC = np.array([cpuC]).transpose()
		cpu = np.concatenate((cpu, cpuC), axis=1)
		alloc = []; prev_alloc = {}
		for c in self.env.containerlist:
			oneHot = [0] * len(self.env.hostlist)
			if c: prev_alloc[c.id] = c.getHostID()
			if c and c.getHostID() != -1: oneHot[c.getHostID()] = 1
			else: oneHot[np.random.randint(0,len(self.env.hostlist))] = 1
			alloc.append(oneHot)
		init = np.concatenate((cpu, alloc), axis=1)
		result, fitness = HGPopt(init, self.model, self.data_type)
		decision = []
		for cid in prev_alloc:
			one_hot = result[cid, -self.hosts:].tolist()
			new_host = one_hot.index(max(one_hot))
			if prev_alloc[cid] != new_host: decision.append((cid, new_host))
		return decision

	def selection(self):
		return []

	def placement(self, containerIDs):
		first_alloc = np.all([not (c and c.getHostID() != -1) for c in self.env.containerlist])
		decision = self.run_HGP()
		return decision