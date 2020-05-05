from .Scheduler import *
import numpy as np
from copy import deepcopy

class LRMMTScheduler(Scheduler):
	def __init__(self):
		super().__init__()
		self.utilHistory = []

	def updateUtilHistory(self):
		hostUtils = []
		for host in self.env.hostlist:
			hostUtils.append(host.getCPU())
		self.utilHistory.append(hostUtils)

	def selection(self):
		self.updateUtilHistory()
		selectedHostIDs = self.RLRSelection(self.utilHistory)
		selectedVMIDs = self.MMTVMSelection(selectedHostIDs)
		return selectedVMIDs

	def placement(self, containerIDs):
		decision = []
		for cid in containerIDs:
			decision.append((cid, np.random.randint(0, len(self.env.hostlist))))
		return decision