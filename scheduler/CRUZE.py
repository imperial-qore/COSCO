from .Scheduler import *
import numpy as np

class CRUZEScheduler(Scheduler):
	def __init__(self, datatype=None):
		super().__init__()
		self.utilHistory = []

	def updateUtilHistory(self):
		hostUtils = []
		for host in self.env.hostlist:
			hostUtils.append(host.getCPU())
		self.utilHistory.append(hostUtils)

	def selection(self):
		self.updateUtilHistory()
		fitnesvals = []
		for i, host in enumerate(self.env.hostlist):
			fitness = host.getCPU() + self.env.stats.metrics[-1]['slaviolationspercentage']
			fitnesvals.append(fitness)
		selectedHostIDs = [np.argmax(fitnesvals)]
		selectedContainerIDs = []
		for hostID in selectedHostIDs:
			containerIDs = self.env.getContainersOfHost(hostID)
			if len(containerIDs):
				containerIPS = [self.env.containerlist[cid].getBaseIPS() for cid in containerIDs]
				selectedContainerIDs.append(containerIDs[containerIPS.index(max(containerIPS))])
		return selectedContainerIDs

	def placement(self, containerIDs):
		decision = []
		hostIPSs = [(self.env.hostlist[i].getCPU(), i) for i in range(len(self.env.hostlist))]
		for cid in containerIDs:
			leastFullHost = min(hostIPSs)
			decision.append((cid, leastFullHost[1]))
			hostIPSs.remove(leastFullHost)
		return decision