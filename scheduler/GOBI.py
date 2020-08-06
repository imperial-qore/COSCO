import sys
sys.path.append('scheduler/BPTI/')

from .Scheduler import *
from .BPTI.train import *

class GOBIScheduler(Scheduler):
	def __init__(self, data_type):
		super().__init__()
		self.model = eval(data_type+"()")
		self.model, _, _, _ = load_model(data_type, self.model, data_type)

	def selection(self):
		selectedHostIDs = self.ThresholdHostSelection()
		selectedVMIDs = self.MMTContainerSelection(selectedHostIDs)
		return selectedVMIDs

	def placement(self, containerIDs):
		return self.RandomPlacement(containerIDs)