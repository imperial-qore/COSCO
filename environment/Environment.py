from host.Host import *
from container.Container import *

class Environment():
	# Total power in watt
	# Total Router Bw
	# Interval Time in seconds
	def __init__(self, TotalPower, RouterBw, Scheduler, ContainerLimit, HostLimit, IntervalTime, hostinit):
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
		self.inactiveContainers = []
		self.addHostlistInit(hostinit)

	def addHostInit(self, IPS, RAM, Disk, Bw, Powermodel):
		assert len(self.hostlist) < self.hostlimit
		host = Host(len(self.hostlist), IPS, RAM, Disk, Bw, Powermodel, self)
		self.hostlist.append(host)

	def addHostlistInit(self, hostList):
		assert len(hostList) == self.hostlimit
		for IPS, RAM, Disk, Bw, Powermodel in hostList:
			self.addHostInit(IPS, RAM, Disk, Bw, Powermodel)

	def addContainerInit(self, CreationID, CreationInterval, IPSModel, RAMModel, DiskModel):
		container = Container(len(self.containerlist), CreationID, CreationInterval, IPSModel, RAMModel, DiskModel, self, HostID = -1)
		self.containerlist.append(container)
		return container

	def addContainerListInit(self, containerInfoList):
		deployed = containerInfoList[:min(len(containerInfoList), self.containerlimit)]
		deployedContainers = []
		for CreationID, CreationInterval, IPSModel, RAMModel, DiskModel in deployed:
			dep = self.addContainerInit(CreationID, CreationInterval, IPSModel, RAMModel, DiskModel)
			deployedContainers.append(dep)
		self.containerlist += [None] * (self.containerlimit - len(self.containerlist))
		return [container.id for container in deployedContainers]

	def addContainer(self, CreationID, CreationInterval, IPSModel, RAMModel, DiskModel):
		container = Container(len(self.containerlist), CreationID, IPSModel, RAMModel, DiskModel, self, HostID = -1)
		for i,c in enumerate(self.containerlist):
			if c == None or not c.active:
				self.containerlist[i] = container
		return container

	def addContainerList(self, containerInfoList):
		deployed = containerInfoList[:min(len(containerInfoList), self.containerlimit)]
		deployedContainers = []
		for CreationID, CreationInterval, IPSModel, RAMModel, DiskModel in deployed:
			dep = self.addContainerInit(CreationID, CreationInterval, IPSModel, RAMModel, DiskModel)
			deployedContainers.append(dep)
		return [container.creationID for container in deployedContainers]

	def getContainersofHost(self, hostID):
		containers = []
		for container in self.containerlist:
			if container.hostID == hostID:
				container.append(container)
		return containers

	def getContainerByID(self, containerID):
		return self.containerlist[containerID]

	def getContainerByCID(self, creationID):
		for c in self.constainerlist + self.inactiveContainers:
			if c and c.creationID == creationID:
				return c

	def getHostByID(self, hostID):
		return self.hostlist[hostID]

	def getCreationIDs(self, containerIDs):
		return [self.containerlist[cid].creationID for cid in containerIDs]

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

	def addContainersInit(self, containerInfoListInit):
		self.interval += 1
		deployed = self.addContainerListInit(containerInfoListInit)
		return deployed

	def allocateInit(self, decision):
		migrated = []
		routerBwToEach = self.totalbw / len(decision)
		for (cid, hid) in decision:
			container = self.getContainerByID(cid)
			assert container.getHostID() == -1
			numberAllocToHost = self.scheduler.getMigrationToHost(hid, decision)
			allocbw = min(self.getHostByID(hid).bwCap.downlink / numberAllocToHost, routerBwToEach)
			if container.getHostID() == hid  or self.getPlacementPossible(cid, hid):
				if container.getHostID() != hid:
					migrated.append((cid, hid))
				container.allocAndExecute(hid, allocbw)
		return migrated

	def destroyCompletedContainers(self):
		destroyed = []
		for i,container in enumerate(self.containerlist):
			if container.getIPS() == 0:
				container.destroy()
				self.containerlist[i] = None
				self.inactiveContainers.append(container)
				destroyed.append(container)

	def getNumActiveContainers(self):
		num = 0 
		for container in self.containerlist:
			if container.active: num += 1
		return num

	def addContainers(self, newContainerList):
		self.interval += 1
		destroyed = self.destroyCompletedContainers()
		deployed = self.addContainerList(newContainerlist)
		return deployed

	def simulationStep(self, newContainerlist, decision):
		routerBwToEach = self.totalbw / len(decision)
		for (cid, hid) in decision:
			assert self.getContainerByID(cid).getHostID() == -1
			currentHostID = self.getContainerByID(cid).getHostID()
			currentHost = self.getHostByID(currentHostID)
			targetHost = self.getHostByID(hid)
			migrateFromNum = self.scheduler.getMigrationFromHost(currentHostID, decision)
			migrateToNum = self.scheduler.getMigrationToHost(hid, decision)
			allocbw = min(targetHost.bwCap.downlink / migrateToNum, currentHost.bwCap.uplink / migrateFromNum, routerBwToEach)
			if container.getHostID() == hid  or self.getPlacementPossible(cid, hid):
				if container.getHostID() != hid:
					migrated.append((cid, hid))
				container.allocAndExecute(hid, allocbw)
		return deployed, mirgated, destroyed