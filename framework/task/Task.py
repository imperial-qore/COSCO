
class Task():
	# IPS = ips requirement
	# RAM = ram requirement in MB
	# Size = container size in MB
	def __init__(self, ID, creationID, creationInterval, IPSModel, RAMModel, DiskModel, application, Environment, HostID = -1):
		self.id = ID
		self.creationID = creationID
		self.ipsmodel = IPSModel
		self.ipsmodel.allocContainer(self)
		self.rammodel = RAMModel
		self.rammodel.allocContainer(self)
		self.diskmodel = DiskModel
		self.diskmodel.allocContainer(self)
		self.hostid = HostID
		self.json_body = {}
		self.env = Environment
		self.createAt = creationInterval
		self.startAt = self.env.interval
		self.totalExecTime = 0
		self.totalMigrationTime = 0
		self.active = True
		self.destroyAt = -1
		self.lastContainerSize = 0
		self.application=application
		self.containerInsert()
		

	
	def containerInsert(self):
		data = []
		image = 'busybox'
		json_body = {
							"measurement":"CreatedContainers",
							"tags": {
										"creation_id":self.creationID,
										"Container_creation_id" :self.id,
										
									},
							"time": self.createAt,
							"fields":
									{
										"Host_id":self.hostid,
										"name":str(self.creationID)+"_"+str(self.id),
										"image":image,
										"Active":self.active,
										"totalExecTime":self.totalExecTime,
										"startAt":self.startAt,
										"createAt":self.createAt,
										"destroyAt":self.destroyAt,
										"IPS_size":self.ipsmodel.constant_ips,
										"IPS_max":self.ipsmodel.max_ips,
										"IPS_duration":self.ipsmodel.duration,
										"IPS_SLA":self.ipsmodel.SLA,
										"RAM_size":self.rammodel.size,
										"RAM_read":self.rammodel.read,
										"RAM_write":self.rammodel.write,
										"DISK_size":self.diskmodel.constant_size,
										"DISK_read":self.diskmodel.constant_read,
										"DISK_write":self.diskmodel.constant_write,
									}
						}
		self.json_body  = json_body
		data.append(json_body)				
		self.env.db.insert(data)	

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
		self.json_body["fields"]["Host_id"]=hostID
		return lastMigrationTime

	def execute(self, lastMigrationTime):
		# Migration time is the time to migrate to new host
		# Thus, execution of task takes place for interval
		# time - migration time with apparent ips
		assert self.hostid != -1
		data = []
		data.append(self.json_body)
		#print(self.json_body)
		self.env.controller.Create(self.json_body)
		self.env.db.insert(data)
		self.totalMigrationTime += lastMigrationTime
		execTime = self.env.intervaltime - lastMigrationTime
		apparentIPS = self.getApparentIPS()
		requiredExecTime = (self.ipsmodel.totalInstructions - self.ipsmodel.completedInstructions) / apparentIPS if apparentIPS else 0
		self.totalExecTime += min(execTime, requiredExecTime)
		self.ipsmodel.completedInstructions += apparentIPS * min(execTime, requiredExecTime)

	def allocateAndExecute(self, hostID, allocBw):
		self.execute(self.allocate(hostID, allocBw))

	def allocateAndrestore(self,hostID,allobw):
		self.json_body["fields"]["Host_id"] = hostID
		self.env.controller.restore(self.json_body)
		data = []
		data.append(self.json_body)
		#print(self.json_body)
		self.env.db.insert(data)
		
	def destroy(self):
		#print("Container destroying process started",self.env.interval)
		rc = self.env.controller.destroy(self.json_body)
	#	print("Response after container destroy",self.json_body["fields"]["name"],self.json_body["fields"]["Host_id"],rc)
		query = "DELETE FROM CreatedContainers WHERE creation_id="+"'"+str(self.creationID)+"'"+";"
	#	print("Deleting from AllocatedContainers")
		self.env.db.delete_measurement(query)
		self.json_body["tags"]["active"] = False
		self.json_body["fields"]["Host_id"] = -1
		self.destroyAt = self.env.interval
		self.hostid = -1
		self.active = False
		
		


