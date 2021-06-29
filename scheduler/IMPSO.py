import sys
sys.path.append('scheduler/BaGTI/')

from .Scheduler import *
from .BaGTI.train import *
from .BaGTI.src.utils import *
from .BaGTI.src.psoW import *

class IMPSOScheduler(Scheduler):
	def __init__(self, data_type):
		super().__init__()
		dtl = data_type.split('_')
		data_type = '_'.join(dtl[:-1])+'W_'+dtl[-1]
		self.model = eval(data_type+"()")
		self.model, _, _, _ = load_model(data_type, self.model, data_type)
		self.data_type = data_type
		self.hosts = int(data_type.split('_')[-1])
		dtl = data_type.split('_')
		self.dataset, _, self.max_container_ips = eval("load_"+'_'.join(dtl[:-1])+"_data("+dtl[-1]+")")

	def run_GA(self):
		cpu = [host.getCPU()/100 for host in self.env.hostlist]
		cpu = np.array([cpu]).transpose()
		cpuC = [(c.getApparentIPS()/self.max_container_ips if c else 0) for c in self.env.containerlist]
		cpuC = np.array([cpuC]).transpose()
		cpu = np.concatenate((cpu, cpuC), axis=1)
		apps, edges = self.env.stats.formGraph()
		apps = torch.tensor(np.array(apps))
		graph = dgl.DGLGraph(edges); 
		if graph.num_nodes() < self.hosts*4:
			graph.add_nodes(self.hosts*4 - graph.num_nodes())
		elif graph.num_nodes() > self.hosts*4:
			graph.remove_nodes(torch.tensor(list(range(self.hosts*4, graph.num_nodes()))))
		graph = dgl.add_self_loop(graph)
		alloc = []; prev_alloc = {}
		for c in self.env.containerlist:
			oneHot = [0] * len(self.env.hostlist)
			if c: prev_alloc[c.id] = c.getHostID()
			if c and c.getHostID() != -1: oneHot[c.getHostID()] = 1
			else: oneHot[np.random.randint(0,len(self.env.hostlist))] = 1
			alloc.append(oneHot)
		init = np.concatenate((cpu, alloc), axis=1)
		init = torch.tensor(init, dtype=torch.float, requires_grad=True)
		result, iteration, fitness = psoW(self.dataset, apps, graph, self.model, [], self.data_type, self.hosts)
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
		decision = self.run_GA()
		return decision