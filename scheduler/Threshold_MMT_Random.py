from .Scheduler import *
import numpy as np
from copy import deepcopy

class TMRScheduler(Scheduler):
	def __init__(self):
		super().__init__()

	def selection(self):
		selectedHostIDs = self.ThresholdHostSelection()
		selectedVMIDs = self.MMTVMSelection(selectedHostIDs)
		return selectedVMIDs

	def placement(self, containerIDs):
		decision = []
		for cid in containerIDs:
			decision.append((cid, np.random.randint(0, len(self.env.hostlist))))
		return decision