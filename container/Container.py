
class Container():
	# IPS = ips requirement
	# RAM = ram requirement in MB
	# Size = container size in MB
	def __init__(self, ID, creationID, IPSModel, RAMModel, DiskModel, Environment, HostID = -1):
		self.id = ID
		self.creationID = creationID
		self.ipsmodel = IPSModel
		self.ipsmodel.allocContainer(self)
		self.rammodel = RAMModel
		self.rammodel.allocContainer(self)
		self.diskmodel = DiskModel
		self.DiskModel.allocContainer(self)
		self.hostid = HostID
		self.env = Environment
		self.totalExecTime = 0
		self.totalMigrationTime = 0
		self.active = True

	def getBaseIPS(self):
		return self.ipsmodel.ips()

	def getApparentIPS(self):
		hostBaseIPS = self.getHost().getBaseIPS()
		hostIPSCap = self.getHost().ipsCap
		canUseIPS = (hostIPSCap - hostBaseIPS) / len(self.env.getContainersofHost(self.hostid))
		return min(self.ipsmodel.max_ips(), self.ipsmodel.ips() + canUseIPS)

	def getRAM(self):
		return self.rammodel.ram()

	def getDisk(self):
		return self.diskmodel.disk()

	def getContainerSize(self):
		return self.rammodel.size

	def getHostID(self):
		return self.hostid

	def getHost(self):
		return self.env.getHostByID(self.hostid)

	def allocate(self, hostID, allocBw):
		lastMigrationTime = self.getContainerSize() / allocBw 
		self.hostid = hostID
		return lastMigrationTime

	def execute(self, lastMigrationTime):
		# Migration time is the time to migrate to new host
		# Thus, execution of task takes place 
		self.totalMigrationTime += lastMigrationTime
		execTime = self.env.intervaltime - lastMigrationTime
		apparentIPS = self.getApparentIPS()
		requiredExecTime = (self.ipsmodel.totalInstruction - self.ipsmodel.completedInstructions) / apparentIPS
		self.totalExecTime += min(execTime, requiredExecTime)
		self.ipsmodel.completedInstructions += apparentIPS * min(execTime, requiredExecTime)

	def allocateAndExecute(self, hostID, allocBw):
		self.execute(self.allocate(hostID, allocBw))

	def destroy(self):
		self.hostID = -1
		self.active = False


