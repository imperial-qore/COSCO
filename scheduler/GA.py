import sys
sys.path.append('scheduler/BaGTI/')

from .Scheduler import *
from .BaGTI.train import *

class GAScheduler(Scheduler):
	def __init__(self, data_type):
		super().__init__()
		self.model = eval(data_type+"()")
		self.model, _, _, _ = load_model(data_type, self.model, data_type)
		self.hosts = int(data_type.split('_')[-1])
		self.data_type = data_type
		dtl = data_type.split('_')
		self.dataset, self.dataset_size, self.max_container_ips = eval("load_"+'_'.join(dtl[:-1])+"_data("+dtl[-1]+")")

	def run_GA(self):
		cpu = [host.getCPU()/100 for host in self.env.hostlist]
		cpu = np.array([cpu]).transpose()
		if 'latency' in self.model.name:
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
		init = torch.tensor(init, dtype=torch.float, requires_grad=True)
		result, iteration, fitness = ga(self.dataset, self.model, [], self.data_type, self.hosts)
		decision = []
		for cid in prev_alloc:
			one_hot = result[cid, (2 if 'latency' in self.model.name else 1):].tolist()
			new_host = one_hot.index(max(one_hot))
			if prev_alloc[cid] != new_host: decision.append((cid, new_host))
		return decision

	def selection(self):
		return []

	def placement(self, containerIDs):
		first_alloc = np.all([not (c and c.getHostID() != -1) for c in self.env.containerlist])
		decision = self.run_GA()
		return decision