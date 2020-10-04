from framework.metrics.Disk import *
from framework.metrics.RAM import *
from framework.metrics.Bandwidth import *

class Task():
	# IPS = ips requirement
	# RAM = ram requirement in MB
	# Size = container size in MB
	def __init__(self, ID, creationID, creationInterval, sla, application, Framework, HostID = -1):
		self.id = ID
		self.creationID = creationID
		# Initial utilization metrics
		self.ips = 0
		self.ram = RAM(0, 0, 0)
		self.bw = Bandwidth(0, 0)
		self.disk = Disk(0, 0, 0)
		self.sla = sla
		self.hostid = HostID
		self.json_body = {}
		self.env = Framework
		self.createAt = creationInterval
		self.startAt = self.env.interval
		self.totalExecTime = 0
		self.totalMigrationTime = 0
		self.active = True
		self.destroyAt = -1
		self.application = application
		self.containerDBInsert()
		
	def containerDBInsert(self):
		self.json_body = {
							"measurement": "CreatedContainers",
							"tags": {
										"container_id": self.id,
										"container_creation_id": self.creationID,
										
									},
							"creation_interval": self.createAt,
							"start_interval": self.startAt,
							"fields":
									{
										"Host_id": self.hostid,
										"name": str(self.creationID)+"_"+str(self.id),
										"image": self.application,
										"Active": self.active,
										"totalExecTime": self.totalExecTime,
										"startAt": self.startAt,
										"createAt": self.createAt,
										"destroyAt": self.destroyAt,
										"IPS": self.ips,
										"SLA": self.sla,
										"RAM_size": self.ram.size,
										"RAM_read": self.ram.read,
										"RAM_write": self.ram.write,
										"DISK_size": self.disk.size,
										"DISK_read": self.disk.read,
										"DISK_write": self.disk.write,
									}
						}
		self.env.db.insert([self.json_body])

	def getBaseIPS(self):
		return self.ips

	def getApparentIPS(self):
		return self.ips

	def getRAM(self):
		return self.ram.size, self.ram.read, self.ram.write

	def getDisk(self):
		return self.disk.size, self.disk.read, self.disk.write

	def getContainerSize(self):
		return self.ram.size

	def getHostID(self):
		return self.hostid

	def getHost(self):
		return self.env.getHostByID(self.hostid)

	# TODO: Update this
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
		self.env.controller.Create(self.json_body)
		self.env.db.insert([self.json_body])
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
		self.env.db.insert([self.json_body])
		
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

	# TODO: Implement this
	def updateUtilizationMetrics(self, json):
		pass
		
		


