from .Scheduler import *
import numpy as np
from copy import deepcopy


class MADMCRScheduler(Scheduler):
    def __init__(self):
        super().__init__()
        self.utilHistory = []

    def updateUtilHistory(self):
        hostUtils = []
        for host in self.env.hostlist:
            hostUtils.append(host.getCPU())
        self.utilHistory.append(hostUtils)

    def selection(self):
        self.updateUtilHistoryContainer()
        selectedHostIDs = self.ThresholdHostSelection()
        selectedVMIDs = self.MaxCorContainerSelection(selectedHostIDs,self.utilHistoryContainer)
        return selectedVMIDs

    def placement(self, containerIDs):
        return self.RandomPlacement(containerIDs)
