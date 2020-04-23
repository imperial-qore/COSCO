import numpy as np

class SWSD():
	def __init__(self, num_worloads):
		self.num_worloads = num_worloads
		self.creation_id = 0

	def generateNewContainers(self):
		workloadlist = []
		for i in range(self.num_worloads):
			CreationID = self.creation_id
			IPSModel = IPSMConstant(np.random.randint(100,150))
			ramMultiplier = np.random.randint(1,5)
			RAMModel = RMConstant(100*ramMultiplier, 50*ramMultiplier, 20*ramMultiplier)
			diskMultiplier = np.random.randint(1,3)
			DiskModel = DMConstant(300*diskMultiplier, 100*diskMultiplier, 120*diskMultiplier)
			workloadlist.append(CreationID, IPSModel, RAMModel, DiskModel)
			self.creation_id += 1
		return workloadlist
