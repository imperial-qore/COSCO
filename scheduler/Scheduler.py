import math
from utils.MathUtils import *
from utils.MathConstants import *

class Scheduler():
    def __init__(self):
        self.env = None

    def setEnvironment(self, env):
        self.env = env

    def selection(self):
        pass

    def placement(self, containerlist):
        pass

    def getMigrationFromHost(self, hostID, decision):
        containerIDs = []
        for (cid, _) in decision:
            hid = self.env.getContainerByID(cid).getHostID()
            if hid == hostID:
                containerIDs.append(cid)
        return containerIDs

    def getMigrationToHost(self, hostID, decision):
        containerIDs = []
        for (cid, hid) in decision:
            if hid == hostID:
                containerIDs.append(cid)
        return containerIDs

    # Host selection

    def ThresholdHostSelection(self):
        selectedHostIDs = []
        for i, host in enumerate(self.env.hostlist):
            if host.getCPU() > 70:
                selectedHostIDs.append(i)
        return selectedHostIDs

    def LRSelection(self, utilHistory):
        if (len(utilHistory) < LOCAL_REGRESSION_BANDWIDTH):
            return self.ThresholdHostSelection()
        selectedHostIDs = []; x = list(range(LOCAL_REGRESSION_BANDWIDTH))
        for i,host in enumerate(self.env.hostlist):
            hostL = [utilHistory[j][i] for j in range(len(utilHistory))]
            _, estimates = loess(x, hostL[-LOCAL_REGRESSION_BANDWIDTH:], poly_degree=1, alpha=0.6)
            weights = estimates['b'].values[-1]
            predictedCPU = weights[0] + weights[1] * (LOCAL_REGRESSION_BANDWIDTH + 1)
            if 1.2 * predictedCPU >= 100:
                selectedHostIDs.append(i)
        return selectedHostIDs

    def RLRSelection(self, utilHistory):
        if (len(utilHistory) < 10):
            return self.ThresholdHostSelection()
        selectedHostIDs = []; x = list(range(LOCAL_REGRESSION_BANDWIDTH))
        for i,host in enumerate(self.env.hostlist):
            hostL = [utilHistory[j][i] for j in range(len(utilHistory))]
            _, estimates = loess(x, hostL[-LOCAL_REGRESSION_BANDWIDTH:], poly_degree=1, alpha=0.6, robustify=True)
            weights = estimates['b'].values[-1]
            predictedCPU = weights[0] + weights[1] * (LOCAL_REGRESSION_BANDWIDTH + 1)
            if 1.2 * predictedCPU >= 100:
                selectedHostIDs.append(i)
        return selectedHostIDs

    # Container Selection

    def MMTVMSelection(self, selectedHostIDs):
        selectedVMIDs = []
        for hostID in selectedHostIDs:
            containerIDs = self.env.getContainersOfHost(hostID)
            ramSize = [self.env.containerlist[cid].getContainerSize() for cid in containerIDs]
            mmtContainerID = containerIDs[ramSize.index(min(ramSize))]
            selectedVMIDs.append(mmtContainerID)
        return selectedVMIDs

    def MaxUseSel(self):
        selectedIDs = []
        for hostID, host in enumerate(self.env.hostlist):
            containerIDs = self.env.getContainersOfHost(hostID)
            if len(containerIDs):
                containerIPS = [self.env.containerlist[cid].getBaseIPS() for cid in containerIDs]
                selectedIDs.append(containerIDs[containerIPS.index(max(containerIPS))])
        return selectedIDs

    # Container placement

    def FirstFitPlacement(self, containerIDs):
        selectedhost = []
        hostlist = self.env.hostlist
        i = 0;
        for cid in containerIDs:
            if len(hostlist) != i:
                selectedhost.append((cid, i))
                i += 1
        return selectedhost

    def LeastFulPlacement(self, containerIDs):
        selectedhost = []
        hostIPS = []
        hosts = self.env.hostlist
        for i, host in enumerate(hosts):
            hostIPS.append(host.getIPSAvailable())
        for cid in containerIDs:
            if len(hostIPS):
                minhost = min(hostIPS)
                selectedhost.append((cid, hostIPS.index(minhost)))
                hostIPS.remove(minhost)
        return selectedhost

    def MaxFulPlacement(self, containerIDs):
        selectedhost = []
        hostIPS = []
        hosts = self.env.hostlist
        for i, host in enumerate(hosts):
            hostIPS.append(host.getIPSAvailable())
        for cid in containerIDs:
            if len(hostIPS):
                minhost = max(hostIPS)
                selectedhost.append((cid, hostIPS.index(minhost)))
                hostIPS.remove(minhost)
        return selectedhost
