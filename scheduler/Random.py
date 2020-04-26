from .Scheduler import *
import numpy as np
from copy import deepcopy

class RandomScheduler(Scheduler):
	def __init__(self):
		super().__init__()

	def selection(self):
		selectableIDs = self.env.getSelectableContainers()
		selectedCount = np.random.randint(0, len(selectableIDs)) + 1
		selectedIDs = []; 
		while len(selectedIDs) < selectedCount:
			idChoice = np.random.choice(selectableIDs)
			if self.env.containerlist[idChoice]:
				selectedIDs.append(idChoice)
				selectableIDs.remove(idChoice)
		return selectedIDs

	def placement(self, containerIDs):
		decision = []
		for cid in containerIDs:
			decision.append((cid, np.random.randint(0, len(self.env.hostlist))))
		return decision