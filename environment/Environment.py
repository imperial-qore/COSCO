from ..host.Host import *
from ..container.Container import *

class Environment():
	# Total power in watt
	# Total Router Bw
	# Interval Time in seconds
	def __init__(self, TotalPower, RouterBw, Scheduler, ContainerLimit, HostLimit, IntervalTime, hostinit, containerinit):
		self.totalpower = TotalPower
		self.totalbw = RouterBw
		self.hostlimit = HostLimit
		self.scheduler = Scheduler
		self.scheduler.setEnvironment(self)
		self.containerlimit = ContainerLimit
		self.hostlist = []
		self.containerlist = []
		self.intervaltime = IntervalTime
		self.interval = 0
		self.addContainerListInit(containerinit)
		self.addHostlistInit(hostlist)

	def addHostInit(self, IPS, RAM, Disk, Bw, Powermodel):
		assert len(self.hostlist) < self.hostlimit
		host = Host(len(self.hostlist), IPS, RAM, Disk, Bw, Powermodel, self)
		self.hostlist.append(host)

	def addHostlistInit(self, hostList):
		assert len(hostList) == self.hostlimit
		for IPS, RAM, Disk, Bw, Powermode in hostList:
			self.addHostInit(IPS, RAM, Disk, Bw, Powermodel)

	def addContainerInit(self, CreationID, IPSModel, RAMModel, DiskModel):
		assert len(self.containerlist) < self.containerlimit
		container = Container(len(self.containerlist), CreationID, IPSModel, RAMModel, DiskModel, self, HostID = -1)
		self.containerlist.append(container)

	def addContainerListInit(self, containerList):
		containerList = containerList[:min(len(containerlist), self.containerlimit)]
		for CreationID, IPSModel, RAMModel, DiskModel in containerList:
			self.addContainerInit(CreationID, IPSModel, RAMModel, DiskModel)
		self.containerlist += [None] * (self.containerlimit - len(self.containerlist))
		return len(containerList)

	def addContainer(self, CreationID, IPSModel, RAMModel, DiskModel):
		assert self.getNumActiveContainers() < self.containerlimit
		container = Container(len(self.containerlist), CreationID, IPSModel, RAMModel, DiskModel, self, HostID = -1)
		for i,c in enumerate(self.containerlist):
			if c == None or not c.active:
				self.containerlist[i] = container

	def addContainerList(self, containerList):
		for CreationID, IPSModel, RAMModel, DiskModel in containerList:
			self.addContainer(CreationID, IPSModel, RAMModel, DiskModel)

	def getContainersofHost(self, hostID):
		containers = []
		for container in self.containerlist:
			if container.hostID == hostID:
				container.append(container)
		return containers

	def getContainerByID(self, constainerID):
		return self.containerlist[containerID]

	def getHostByID(self, hostID):
		return self.hostlist[hostID]

	def getSelection(self):
		return self.scheduler.selection()

	def getPlacement(self, containerlist):
		return self.scheduler.placement(containerlist)

	def getPlacementPossible(self, containerID, hostID):
		container = self.containerlist[containerID]
		host = self.hostlist[hostID]
		ipsreq = container.getIPS()
		ramsizereq, ramreadreq, ramwritereq = container.getRAM()
		disksizereq, diskreadreq, diskwritereq = containerID.getDisk()
		ipsavailable = host.getIPSAvailable()
		ramsizeav, ramreadav, ramwriteav = host.getRAMAvailable()
		disksizeav, diskreadav, diskwwriteav = host.getDiskAvailable()
		return (ipsreq <= ipsavailable and \
				ramsizereq <= ramsizeav and \
				ramreadreq <= ramreadav and \
				ramwritereq <= ramwriteav and \
				disksizereq <= disksizeav and \
				diskreadreq <= diskreadav and \
				diskwritereq <= diskwriteav)

	def allocateInit(self):
		self.interval += 1
		decision = self.getPlacement(self.containerlist)
		routerBwToEach = self.totalbw / len(decision)
		for (cid, hid) in decision:
			assert self.getContainerByID(cid).getHostID() == -1
			allocbw = min(self.getHostByID(hid).bwCap.downlink, routerBwToEach)
			container.allocate(hid, allocbw)

	def destroyCompletedContainers(self):
		for container in self.containerlist:
			if container.getIPS() == 0:
				container.destroy()

	def getNumActiveContainers(self):
		num = 0 
		for container in self.containerlist:
			if container.active: num += 1
		return num

	def simulationStep(self, newContainerlist):
		self.interval += 1
		self.destroyCompletedContainers()
		self.addContainerList(newContainerlist)
		selectedContainers = self.getSelection()
		decision = self.getPlacement(selectedContainers)
		routerBwToEach = self.totalbw / len(decision)
		for (cid, hid) in decision:
			assert self.getContainerByID(cid).getHostID() == -1
			currentHostID = self.getContainerByID(cid).getHostID()
			currentHost = self.getHostByID(currentHostID)
			targetHost = self.getHostByID(hid)
			allocbw = min(targetHost.bwCap.downlink, currentHost.bwCap.uplink, routerBwToEach)
			container.allocate(hid, allocbw)