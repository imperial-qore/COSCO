import numpy as np

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
			IPSModel = IPSMConstant(1000*ipsMultiplier, 1500*ipsMultiplier, 5*ipsMultiplier, interval + 4*ipsMultiplier)
			ramMultiplier = np.random.randint(1,5)
			RAMModel = RMConstant(100*ramMultiplier, 50*ramMultiplier, 20*ramMultiplier)
			diskMultiplier = np.random.randint(1,3)
			DiskModel = DMConstant(300*diskMultiplier, 100*diskMultiplier, 120*diskMultiplier)
			workloadlist.append(CreationID, IPSModel, RAMModel, DiskModel)
			self.creation_id += 1
		self.createdContainers += workloadlist
		self.deployedContainers += [False] * len(workloadlist)
		return self.getUndeployedContainers()

	def getUndeployedContainers(self):
		undeployed = []
		for container in self.createdContainers:
			if not self.deployedContainers[container.creationID]:
				undeployed.append(container)
		return undeployed

	def updateDeployedContainers(creationIDs):
		for cid in creationIDs:
			assert not self.deployedContainers[cid]
			self.deployedContainers[cid] = True
