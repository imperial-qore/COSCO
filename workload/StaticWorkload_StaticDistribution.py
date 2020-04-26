import numpy as np
from container.IPSModels.IPSMConstant import *
from container.RAMModels.RMConstant import *
from container.DiskModels.DMConstant import *

class SWSD():
	def __init__(self, num_worloads):
		self.num_worloads = num_worloads
		self.creation_id = 0
		self.createdContainers = []
		self.deployedContainers = []

	def generateNewContainers(self, interval):
		workloadlist = []
		for i in range(self.num_worloads):
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

	def getUndeployedContainers(self):
		undeployed = []
		for i,deployed in enumerate(self.deployedContainers):
			if not deployed:
				undeployed.append(self.createdContainers[i])
		return undeployed

	def updateDeployedContainers(self, creationIDs):
		for cid in creationIDs:
			assert not self.deployedContainers[cid]
			self.deployedContainers[cid] = True
