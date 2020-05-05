from .Scheduler import *
import numpy as np
from copy import deepcopy

class RandomScheduler(Scheduler):
	def __init__(self):
		super().__init__()

	def selection(self):
		return self.RandomContainerSelection()

	def placement(self, containerIDs):
		return self.RandomPlacement(containerIDs)