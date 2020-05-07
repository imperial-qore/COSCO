from .Scheduler import *
import numpy as np
from copy import deepcopy


class TMCRScheduler(Scheduler):
    def __init__(self):
        super().__init__()
        self.utilHistory = []
        self.utilHistoryContainer= []


    def updateUtilHistory(self):
        hostUtils = []
        for host in self.env.hostlist:
            hostUtils.append(host.getCPU())
        self.utilHistory.append(hostUtils)

    def updateUtilHistoryContainer(self):
        containerUtil = [(cid.getBaseIPS() if cid else 0) for cid in self.env.containerlist]
        self.utilHistoryContainer.append(containerUtil)


    def selection(self):
        self.updateUtilHistory()
        self.updateUtilHistoryContainer()
        selectedHostIDs = self.ThresholdHostSelection()
        selectedVMIDs = self.MaxCorContainerSelection(selectedHostIDs,self.utilHistoryContainer)
        return selectedVMIDs

    def placement(self, containerIDs):
        return self.RandomPlacement(containerIDs)
