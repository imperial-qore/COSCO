import math
from localreg import *


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

    def getMaxContMigrationTime(self,host):
        containerIDs = self.env.getContainersOfHost(host)
        if len(containerIDs):
            ramSize = [self.env.containerlist[cid].getContainerSize() for cid in containerIDs]
            maxSize = containerIDs[ramSize.index(max(ramSize))]
        else:
            maxSize=0
        return maxSize

    def getLR(self,hostutil,host):
        hostutil.reverse()
        finalUtil=0
        x=[]
        for i in range(len(hostutil[:10])):
            x.append(i+1)
        estimates = localreg(np.array(x), np.array(hostutil[:10]), degree=0, kernel=tricube, width=0.3)
        if len(estimates)>=2:
             maxMigrationTime = math.ceil((self.getMaxContMigrationTime(host)/self.env.intervaltime))
             predicted = estimates[0] + estimates[1] * (len(hostutil[:10]) + maxMigrationTime);
             finalUtil= 1.2*predicted
        return True if finalUtil >= 1 else False

    def getOverloadededHosts(self,utils):
        selectedId=[]
        hostL = []
        for i,host in enumerate(self.env.hostlist):
            for j in range(len(utils)):
                hostL.append(utils[j][i])
            if bool(self.getLR(hostL,i)):
                selectedId.append(i)
        return selectedId

    def LRSelection(self, utilHistory):
        selectedhost=self.getOverloadededHosts(utilHistory)
        for hostid in selectedhost:
            containId=self.env.getContainersOfHost(hostid)
        return selectedhost

    # Container Selection

    def MMTVMSelection(self, selectedHostIDs):
        print(selectedHostIDs)
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
