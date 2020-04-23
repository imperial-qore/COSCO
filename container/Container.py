
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

	def getIPS(self):
		return self.ipsmodel.ips()

	def getRAM(self):
		return self.rammodel.ram()

	def getDisk(self):
		return self.diskmodel.disk()

	def getContainerSize(self):
		return self.rammodel.size

	def getHostID(self):
		return self.hostid

	def allocate(self, hostID, allocBw):
		self.totalMigrationTime += self.getContainerSize() / allocBw 
		self.totalExecTime = self.env.intervaltime
		self.hostid = hostID

	def destroy(self):
		self.hostID = -1
		self.active = False


