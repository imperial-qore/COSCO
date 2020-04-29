import math


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

    def ThresholdHostSelection(self):
        selectedHostIDs = []
        for i, host in enumerate(self.env.hostlist):
            if host.getCPU() > 70:
                selectedHostIDs.append(i)
        return selectedHostIDs

    def MMTVMSelection(self, selectedHostIDs):
        selectedVMIDs = []
        for hostID in selectedHostIDs:
            containerIDs = self.env.getContainersOfHost(hostID)
            ramSize = [self.env.containerlist[cid].getContainerSize() for cid in containerIDs]
            mmtContainerID = containerIDs[ramSize.index(min(ramSize))]
            selectedVMIDs.append(mmtContainerID)
        return selectedVMIDs

    def MaxFulSel(self):
        selectedIDs = []
        for hostID,host in enumerate(self.env.hostlist):
            containerIDs = self.env.getContainersOfHost(hostID)
            if len(containerIDs):
                containerIPS = [self.env.containerlist[cid].getBaseIPS() for cid in containerIDs]
                selectedIDs.append(containerIDs[containerIPS.index(max(containerIPS))])
        return selectedIDs

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
