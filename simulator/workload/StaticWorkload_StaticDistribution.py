from .Workload import *
from simulator.container.IPSModels.IPSMBitbrain import *
from simulator.container.RAMModels.RMBitbrain import *
from simulator.container.DiskModels.DMBitbrain import *

class SWSD(Workload):
	def __init__(self, num_workloads):
		super().__init__()
		self.num_workloads = num_workloads

	def generateNewContainers(self, interval):
		workloadlist = []
		for i in range(self.num_workloads):
			CreationID = self.creation_id
			ipsMultiplier = np.random.randint(1,5)
			IPSModel = IPSMConstant(1000*ipsMultiplier, 1500*ipsMultiplier, 4*ipsMultiplier, interval + 3*ipsMultiplier)
			ramMultiplier = np.random.randint(1,5)
			RAMModel = RMConstant(100*ramMultiplier, 50*ramMultiplier, 20*ramMultiplier)
			diskMultiplier = np.random.randint(1,3)
			DiskModel = DMConstant(300*diskMultiplier, 100*diskMultiplier, 120*diskMultiplier)
			workloadlist.append((CreationID, interval, IPSModel, RAMModel, DiskModel))
			self.creation_id += 1
		self.createdContainers += workloadlist
		self.deployedContainers += [False] * len(workloadlist)
		return self.getUndeployedContainers()