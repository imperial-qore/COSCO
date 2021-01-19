import sys
sys.path.append('scheduler/BaGTI/')

from .Scheduler import *
from .BaGTI.train import *
from .BaGTI.src.utils import *

class HSOGOBI2Scheduler(Scheduler):
	def __init__(self, data_type):
		super().__init__()
		data_type = 'stochastic_' + data_type
		dtl = data_type.split('_')
		data_type = '_'.join(dtl[:-1])+'2_'+dtl[-1]
		self.model = eval(data_type+"()")
		self.model, _, _, _ = load_model(data_type, self.model, data_type)
		self.data_type = data_type
		self.hosts = int(data_type.split('_')[-1])
		_, _, (self.max_container_ips, self.max_energy, self.max_response) = eval("load_"+'_'.join(dtl[:-1])+"2_data("+dtl[-1]+")")

	def run_HSOGOBI2(self):
		cpu = [host.getCPU()/100 for host in self.env.hostlist]
		cpu = np.array([cpu]).transpose()
		if 'latency' in self.model.name:
			cpuC = [(c.getApparentIPS()/self.max_container_ips if c else 0) for c in self.env.containerlist]
			cpuC = np.array([cpuC]).transpose()
			e, r = (0, 0) if self.env.stats == None else self.env.stats.runSimulationGOBI()
			pred = np.broadcast_to(np.array([e/self.max_energy, r/self.max_response]), (self.hosts, 2))
			cpu = np.concatenate((cpu, cpuC, pred), axis=1)
		alloc = []; prev_alloc = {}
		for c in self.env.containerlist:
			oneHot = [0] * len(self.env.hostlist)
			if c: prev_alloc[c.id] = c.getHostID()
			if c and c.getHostID() != -1: oneHot[c.getHostID()] = 1
			else: oneHot[np.random.randint(0,len(self.env.hostlist))] = 1
			alloc.append(oneHot)
		init = np.concatenate((cpu, alloc), axis=1)
		init = torch.tensor(init, dtype=torch.float, requires_grad=True)
		result, iteration, fitness = so_opt(init, self.model, [], self.data_type)
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
		decision = self.run_HSOGOBI2()
		return decision