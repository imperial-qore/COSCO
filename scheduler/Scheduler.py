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
            if LOCAL_REGRESSION_CPU_MULTIPLIER * predictedCPU >= 100:
                selectedHostIDs.append(i)
        return selectedHostIDs

    def RLRSelection(self, utilHistory):
        if (len(utilHistory) < LOCAL_REGRESSION_BANDWIDTH):
            return self.ThresholdHostSelection()
        selectedHostIDs = []; x = list(range(LOCAL_REGRESSION_BANDWIDTH))
        for i,host in enumerate(self.env.hostlist):
            hostL = [utilHistory[j][i] for j in range(len(utilHistory))]
            _, estimates = loess(x, hostL[-LOCAL_REGRESSION_BANDWIDTH:], poly_degree=1, alpha=0.6, robustify=True)
            weights = estimates['b'].values[-1]
            predictedCPU = weights[0] + weights[1] * (LOCAL_REGRESSION_BANDWIDTH + 1)
            if LOCAL_REGRESSION_CPU_MULTIPLIER * predictedCPU >= 100:
                selectedHostIDs.append(i)
        return selectedHostIDs

    # Container Selection

    def RandomContainerSelection(self):
        selectableIDs = self.env.getSelectableContainers()
        selectedCount = np.random.randint(0, len(selectableIDs)) + 1
        selectedIDs = []; 
        while len(selectedIDs) < selectedCount:
            idChoice = np.random.choice(selectableIDs)
            if self.env.containerlist[idChoice]:
                selectedIDs.append(idChoice)
                selectableIDs.remove(idChoice)
        return selectedIDs

    def MMTContainerSelection(self, selectedHostIDs):
        selectedContainerIDs = []
        for hostID in selectedHostIDs:
            containerIDs = self.env.getContainersOfHost(hostID)
            ramSize = [self.env.containerlist[cid].getContainerSize() for cid in containerIDs]
            mmtContainerID = containerIDs[ramSize.index(min(ramSize))]
            selectedContainerIDs.append(mmtContainerID)
        return selectedContainerIDs

    def MaxUseContainerSelection(self, selectedHostIDs):
        selectedContainerIDs = []
        for hostID in selectedHostIDs:
            containerIDs = self.env.getContainersOfHost(hostID)
            if len(containerIDs):
                containerIPS = [self.env.containerlist[cid].getBaseIPS() for cid in containerIDs]
                selectedContainerIDs.append(containerIDs[containerIPS.index(max(containerIPS))])
        return selectedContainerIDs

    # Container placement

    def RandomPlacement(self, containerIDs):
        decision = []
        for cid in containerIDs:
            decision.append((cid, np.random.randint(0, len(self.env.hostlist))))
        return decision

    def FirstFitPlacement(self, containerIDs):
        decision = []
        for cid in containerIDs:
            for hostID in range(len(self.env.hostlist)):
                if self.env.getPlacementPossible(cid, hostID):
                    decision.append((cid, hostID)); break
        return decision

    def LeastFullPlacement(self, containerIDs):
        decision = []
        hostIPSs = [(self.env.hostlist[i].getCPU(), i) for i in range(len(self.env.hostlist))]
        for cid in containerIDs:
            leastFullHost = min(hostIPSs)
            decision.append((cid, leastFullHost[1]))
            hostIPSs.remove(leastFullHost)
        return decision

    def MaxFullPlacement(self, containerIDs):
        decision = []
        hostIPSs = [(self.env.hostlist[i].getCPU(), i) for i in range(len(self.env.hostlist))]
        for cid in containerIDs:
            leastFullHost = min(hostIPSs)
            decision.append((cid, leastFullHost[1]))
            hostIPSs.remove(leastFullHost)
        return decision