
class Container():
	# IPS = ips requirement
	# RAM = ram requirement in MB
	# Size = container size in MB
	def __init__(self, ID, creationID, creationInterval, IPSModel, RAMModel, DiskModel, Environment, HostID = -1):
		self.id = ID
		self.creationID = creationID
		self.ipsmodel = IPSModel
		self.ipsmodel.allocContainer(self)
		self.rammodel = RAMModel
		self.rammodel.allocContainer(self)
		self.diskmodel = DiskModel
		self.diskmodel.allocContainer(self)
		self.hostid = HostID
		self.env = Environment
		self.createAt = creationInterval
		self.startAt = self.env.interval
		self.totalExecTime = 0
		self.totalMigrationTime = 0
		self.active = True
		self.destroyAt = -1
		self.lastContainerSize = 0

	def getBaseIPS(self):
		return self.ipsmodel.getIPS()

	def getApparentIPS(self):
		hostBaseIPS = self.getHost().getBaseIPS()
		hostIPSCap = self.getHost().ipsCap
		canUseIPS = (hostIPSCap - hostBaseIPS) / len(self.env.getContainersOfHost(self.hostid))
		return min(self.ipsmodel.getMaxIPS(), self.getBaseIPS() + canUseIPS)

	def getRAM(self):
		rsize, rread, rwrite = self.rammodel.ram()
		self.lastContainerSize = rsize
		return rsize, rread, rwrite

	def getDisk(self):
		return self.diskmodel.disk()

	def getContainerSize(self):
		if self.lastContainerSize == 0: self.getRAM()
		return self.lastContainerSize

	def getHostID(self):
		return self.hostid

	def getHost(self):
		return self.env.getHostByID(self.hostid)

	def allocate(self, hostID, allocBw):
		# Migrate if different host
		lastMigrationTime = self.getContainerSize() / allocBw if self.hostid != hostID else 0
		self.hostid = hostID
		return lastMigrationTime

	def execute(self, lastMigrationTime):
		# Migration time is the time to migrate to new host
		# Thus, execution of task takes place for interval
		# time - migration time with apparent ips
		assert self.hostid != -1
		self.totalMigrationTime += lastMigrationTime
		execTime = self.env.intervaltime - lastMigrationTime
		apparentIPS = self.getApparentIPS()
		requiredExecTime = (self.ipsmodel.totalInstructions - self.ipsmodel.completedInstructions) / apparentIPS if apparentIPS else 0
		self.totalExecTime += min(execTime, requiredExecTime)
		self.ipsmodel.completedInstructions += apparentIPS * min(execTime, requiredExecTime)

	def allocateAndExecute(self, hostID, allocBw):
		self.execute(self.allocate(hostID, allocBw))

	def destroy(self):
		self.destroyAt = self.env.interval
		self.hostid = -1
		self.active = False


