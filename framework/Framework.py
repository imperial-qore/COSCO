from framework.node.Node import *
from framework.task.Task import *
from framework.server.controller import *

class Framework():
	# Total power in watt
	# Total Router Bw
	# Interval Time in seconds
	def __init__(self, TotalPower, RouterBw, Scheduler, ContainerLimit, HostLimit, IntervalTime, hostinit, database):
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
		self.db = database
		self.inactiveContainers = []
		self.controller= RequestHandler(self.db)
		self.addHostlistInit(hostinit)
	
	def addHostInit(self, IP,IPS, RAM, Disk, Bw, Powermodel):
		assert len(self.hostlist) < self.hostlimit

		host = Host(len(self.hostlist),IP,IPS, RAM, Disk, Bw, Powermodel, self)
		self.hostlist.append(host)

	def addHostlistInit(self, hostList):
		assert len(hostList) == self.hostlimit
		for IP,IPS, RAM, Disk, Bw, Powermodel in hostList:
			self.addHostInit(IP,IPS, RAM, Disk, Bw, Powermodel)

	def addContainerInit(self, CreationID, CreationInterval, IPSModel, RAMModel, DiskModel,application):
		container = Container(len(self.containerlist), CreationID, CreationInterval, IPSModel, RAMModel, DiskModel, application,self, HostID = -1)
		self.containerlist.append(container)

		return container

	def addContainerListInit(self, containerInfoList):
		deployed = containerInfoList[:min(len(containerInfoList), self.containerlimit-self.getNumActiveContainers())]
		deployedContainers = []
		for CreationID, CreationInterval, IPSModel, RAMModel, DiskModel,application in deployed:
			dep = self.addContainerInit(CreationID, CreationInterval, IPSModel, RAMModel, DiskModel,application)
			deployedContainers.append(dep)
		self.containerlist += [None] * (self.containerlimit - len(self.containerlist))
		print(self.containerlist)
		return [container.id for container in deployedContainers]

	def addContainer(self, CreationID, CreationInterval, IPSModel, RAMModel, DiskModel,application):
		for i,c in enumerate(self.containerlist):
			if c == None or not c.active:
				container = Container(i, CreationID, CreationInterval, IPSModel, RAMModel, DiskModel, application,self, HostID = -1)
				self.containerlist[i] = container
				return container

	def addContainerList(self, containerInfoList):
		deployed = containerInfoList[:min(len(containerInfoList), self.containerlimit-self.getNumActiveContainers())]
		deployedContainers = []
		for CreationID, CreationInterval, IPSModel, RAMModel, DiskModel,application in deployed:
			dep = self.addContainer(CreationID, CreationInterval, IPSModel, RAMModel, DiskModel,application)
			deployedContainers.append(dep)
		return [container.id for container in deployedContainers]

	def getContainersOfHost(self, hostID):
		containers = []
		for container in self.containerlist:
			if container and container.hostid == hostID:
				containers.append(container.id)
		return containers

	def getContainerByID(self, containerID):
		return self.containerlist[containerID]

	def getContainerByCID(self, creationID):
		for c in self.containerlist + self.inactiveContainers:
			if c and c.creationID == creationID:
				return c

	def getHostByID(self, hostID):
		return self.hostlist[hostID]

	def getCreationIDs(self, migrations, containerIDs):
		creationIDs = []
		for decision in migrations:
			if decision[0] in containerIDs: creationIDs.append(self.containerlist[decision[0]].creationID)
		return creationIDs

	def getPlacementPossible(self, containerID, hostID):
		container = self.containerlist[containerID]
		host = self.hostlist[hostID]
		ipsreq = container.getBaseIPS()
		ramsizereq, ramreadreq, ramwritereq = container.getRAM()
		disksizereq, diskreadreq, diskwritereq = container.getDisk()
		ipsavailable = host.getIPSAvailable()
		ramsizeav, ramreadav, ramwriteav = host.getRAMAvailable()
		disksizeav, diskreadav, diskwriteav = host.getDiskAvailable()
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
		migrations = []
		routerBwToEach = self.totalbw / len(decision)
		for (cid, hid) in decision:
			container = self.getContainerByID(cid)
			assert container.getHostID() == -1
			numberAllocToHost = len(self.scheduler.getMigrationToHost(hid, decision))
			allocbw = min(self.getHostByID(hid).bwCap.downlink / numberAllocToHost, routerBwToEach)
			if self.getPlacementPossible(cid, hid):
				if container.getHostID() != hid:
					migrations.append((cid, hid))
				container.allocateAndExecute(hid, allocbw)
			# destroy pointer to this unallocated container as book-keeping is done by workload model
			else: 
				self.containerlist[cid] = None
		return migrations

	def destroyCompletedContainers(self):
		destroyed = []
		for i,container in enumerate(self.containerlist):
			if container and container.getBaseIPS() == 0:
				container.destroy()
				self.containerlist[i] = None
				self.inactiveContainers.append(container)
				destroyed.append(container)
		return destroyed

	def getNumActiveContainers(self):
		num = 0 
		for container in self.containerlist:
			if container and container.active: num += 1
		return num

	def getSelectableContainers(self):
		selectable = []
		selected = []
		containers = self.db.select("SELECT * FROM CreatedContainers;")
		for container in self.containerlist:
			if container and container.active and container.getHostID() != -1:
				selectable.append(container.id)
		print(selectable)
		return selectable

	def addContainers(self, newContainerList):
		self.interval += 1
		destroyed = self.destroyCompletedContainers()
		deployed = self.addContainerList(newContainerList)
		return deployed, destroyed

	def getActiveContainerList(self):
		return [c.getHostID() if c and c.active else -1 for c in self.containerlist]

	def getContainersInHosts(self):
		return [len(self.getContainersOfHost(host)) for host in range(self.hostlimit)]

	def simulationStep(self, decision):
		routerBwToEach = self.totalbw / len(decision) if len(decision) > 0 else self.totalbw
		migrations = []
		containerIDsAllocated = []
		for (cid, hid) in decision:
			container = self.getContainerByID(cid)
			currentHostID = self.getContainerByID(cid).getHostID()
			currentHost = self.getHostByID(currentHostID)
			targetHost = self.getHostByID(hid)
			if self.containerlist[cid].hostid != -1:
				self.controller.checkpoint(container.creationID,currentHostID)
			#	print(container.id,currentHostID)
			# Check point and restore --> migrate
			migrateFromNum = len(self.scheduler.getMigrationFromHost(currentHostID, decision))
			migrateToNum = len(self.scheduler.getMigrationToHost(hid, decision))
			allocbw = min(targetHost.bwCap.downlink / migrateToNum, currentHost.bwCap.uplink / migrateFromNum, routerBwToEach)
			# Check point and restore --> migrate 
			if hid != self.containerlist[cid].hostid and self.getPlacementPossible(cid, hid):
				if self.containerlist[cid].hostid != -1:
					print("Migration in process")
	   				#self.controller.checkpoint(container.creationID,currentHostID)
					self.controller.migrate(currentHostID,hid,container.id)
					container.allocateAndrestore(hid, allocbw)
					#container.allocateAndExecute(hid,allocbw)
					containerIDsAllocated.append(cid)
					migrations.append((cid, hid))
				#container.allocateAndrestore(hid, allocbw)
				
		# destroy pointer to unallocated containers as book-keeping is done by workload model
		for (cid, hid) in decision:
			if self.containerlist[cid].hostid == -1: self.containerlist[cid] = None
			
		for i,container in enumerate(self.containerlist):
			if container and i not in containerIDsAllocated:
				container.execute(0)
		return migrations