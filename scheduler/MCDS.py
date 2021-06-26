import sys
sys.path.append('scheduler/BaGTI/')

from .Scheduler import *
from .BaGTI.train import *
from .BaGTI.src.utils import *
from .MCS.MonteCarloTree import *

class MCDSScheduler(Scheduler):
	def __init__(self, data_type):
		super().__init__()
		dtl = data_type.split('_')
		data_type = '_'.join(dtl[:-1])+'W_'+dtl[-1]
		self.model = eval(data_type+"()")
		self.model, _, _, _ = load_model(data_type, self.model, data_type)
		self.data_type = data_type
		self.hosts = int(data_type.split('_')[-1])
		dtl = data_type.split('_')
		_, _, self.max_container_ips = eval("load_"+'_'.join(dtl[:-1])+"_data("+dtl[-1]+")")
		self.mct = MonteCarloTree(expansionSize = 4)

	def run_MCSearch(self, containerIDs):
		apps, edges = self.env.stats.formGraph()
		apps = torch.tensor(np.array(apps))
		graph = dgl.DGLGraph(edges); 
		if graph.num_nodes() < self.hosts*4:
			graph.add_nodes(self.hosts*4 - graph.num_nodes())
		elif graph.num_nodes() > self.hosts*4:
			graph.remove_nodes(torch.tensor(list(range(self.hosts*4, graph.num_nodes()))))
		graph = dgl.add_self_loop(graph)
		senv = SimpleEnv(self.env, apps, graph, self.model, self.max_container_ips)
		self.mct.setEnv(senv)
		decision = self.mct.runSimulations(numSim = 10, containerIDs = containerIDs)
		return decision

	def selection(self):
		return []

	def placement(self, containerIDs):
		first_alloc = np.all([not (c and c.getHostID() != -1) for c in self.env.containerlist])
		decision = self.run_MCSearch(containerIDs)
		return decision