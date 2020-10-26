import math
from utils.MathUtils import *
from utils.MathConstants import *
import pandas as pd
from statistics import median
import numpy as np

class Scheduler():
    def __init__(self):
        self.env = None

    def setEnvironment(self, env):
        self.env = env

    def selection(self):
        pass

    def placement(self, containerlist):
        pass

    def filter_placement(self, decision):
        filtered_decision = []
        for cid, hid in decision:
            if self.env.getContainerByID(cid).getHostID() != hid:
                filtered_decision.append((cid, hid))
        return filtered_decision

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

    def MADSelection(self, utilHistory):
        selectedHostIDs = []
        for i, host in enumerate(self.env.hostlist):
            hostL = [utilHistory[j][i] for j in range(len(utilHistory))]
            median_hostL = np.median(np.array(hostL))
            mad = np.median([abs(Utilhst-median_hostL) for Utilhst in hostL])
            ThresholdCPU = 100-LOCAL_REGRESSION_CPU_MULTIPLIER * mad
            UtilizedCPU = host.getCPU()
            if UtilizedCPU > ThresholdCPU:
                selectedHostIDs.append(i)
        return selectedHostIDs

    def IQRSelection(self, utilHistory):
        selectedHostIDs = []
        for i, host in enumerate(self.env.hostlist):
            hostL = [utilHistory[j][i] for j in range(len(utilHistory))]
            q1, q3 = np.percentile(np.array(hostL), [25, 75])
            IQR = q3-q1
            ThresholdCPU = 100-LOCAL_REGRESSION_CPU_MULTIPLIER * IQR
            UtilizedCPU = host.getCPU()
            if UtilizedCPU > ThresholdCPU:
                selectedHostIDs.append(i)
        return selectedHostIDs

    # Container Selection

    def RandomContainerSelection(self):
        selectableIDs = self.env.getSelectableContainers()
        if selectableIDs == []: return []
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
            if ramSize:
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

    def MaxCorContainerSelection(self, selectedHostIDs,utilHistoryContainer):
        selectedContainerIDs = []
        for hostID in selectedHostIDs:
            containerIDs = self.env.getContainersOfHost(hostID)
            if len(containerIDs):
                hostL = [[utilHistoryContainer[j][cid] for j in range(len(utilHistoryContainer))] for cid in containerIDs]
                data = pd.DataFrame(hostL)
                data = data.T; RSquared = []
                for i in range(data.shape[1]):
                    x = np.array(data.drop(data.columns[i],axis=1))
                    y = np.array(data.iloc[:,i])
                    X1 = np.c_[x, np.ones(x.shape[0])]
                    y_pred = np.dot(X1, np.dot(np.linalg.pinv(np.dot(np.transpose(X1), X1)), np.dot(np.transpose(X1), y)))
                    corr = np.corrcoef(np.column_stack((y,y_pred)), rowvar=False)
                    RSquared.append(corr[0][1] if not np.isnan(corr).any() else 0)
                selectedContainerIDs.append(containerIDs[RSquared.index(max(RSquared))])
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