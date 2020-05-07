from .Scheduler import *

class IQRMMTRScheduler(Scheduler):
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
        selectedHostIDs = self.IQRSelection(self.utilHistory)
        selectedVMIDs = self.MMTContainerSelection(selectedHostIDs)
        return selectedVMIDs

    def placement(self, containerIDs):
        return self.RandomPlacement(containerIDs)
