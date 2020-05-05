from .Scheduler import *
import numpy as np
from copy import deepcopy


class TMRScheduler(Scheduler):
    def __init__(self):
        super().__init__()

    def selection(self):
        selectedHostIDs = self.ThresholdHostSelection()
        selectedIDs = self.MaxUseContainerSelection(selectedHostIDs)
        return selectedIDs

    def placement(self, containerIDs):
        return self.RandomPlacement(containerIDs)
