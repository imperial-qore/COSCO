from .Scheduler import *
import numpy as np
from copy import deepcopy

class RandomScheduler(Scheduler):
	def __init__(self):
		super().__init__()

	def selection(self):
		containerlist = self.env.containerlist
		selectedCount = np.random.randint(0, self.env.getNumActiveContainers()) + 1
		selectedIDs = []; allIDs = list(range(len(containerlist)))
		while len(selectedIDs) < selectedCount:
			idChoice = np.random.choice(allIDs)
			if self.env.containerlist[idChoice]:
				selectedIDs.append(idChoice)
				allIDs.remove(idChoice)
		return selectedIDs

	def placement(self, containerIDs):
		decision = []
		for cid in containerIDs:
			decision.append((cid, np.random.randint(0, len(self.env.hostlist))))
		return decision