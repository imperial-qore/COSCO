from metrics.Disk import *
from metrics.RAM import *
from metrics.Bandwidth import *
from dateutil import parser
from datetime import datetime

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
		self.lastReadBytes = 0
		self.lastWriteBytes = 0
		self.totalExecTime = 0
		self.totalMigrationTime = 0
		self.active = True
		self.destroyAt = -1
		self.application = application
		self.execError = ""
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

	def allocateAndExecute(self, hostID):
		# self.env.logger.debug("Allocating container "+self.json_body['fields']['name']+" to host "+self.env.getHostByID(hostID).ip)
		self.hostid = hostID
		self.json_body["fields"]["Host_id"] = hostID
		_, lastMigrationTime = self.env.controller.create(self.json_body, self.env.getHostByID(self.hostid).ip)
		self.totalMigrationTime += lastMigrationTime
		execTime = self.env.intervaltime - lastMigrationTime
		self.totalExecTime += execTime

	def allocateAndrestore(self, hostID):
		# self.env.logger.debug("Migrating container "+self.json_body['fields']['name']+" from host "+self.getHost().ip+
		# 	" to host "+self.env.getHostByID(hostID).ip)
		cur_host_ip = self.getHost().ip
		self.hostid = hostID
		tar_host_ip = self.getHost().ip
		self.json_body["fields"]["Host_id"] = hostID
		_, checkpointTime = self.env.controller.checkpoint(self.creationID, self.id, cur_host_ip)
		_, migrationTime = self.env.controller.migrate(self.creationID, self.id, cur_host_ip, tar_host_ip)
		_, restoreTime = self.env.controller.restore(self.creationID, self.id, self.application, tar_host_ip)
		lastMigrationTime = checkpointTime + migrationTime + restoreTime
		self.totalMigrationTime += lastMigrationTime
		execTime = self.env.intervaltime - lastMigrationTime
		self.totalExecTime += execTime
		
	def destroy(self):
		assert not self.active
		rc = self.env.controller.destroy(self.json_body, self.getHost().ip)
		# query = "DELETE FROM CreatedContainers WHERE creation_id="+"'"+str(self.creationID)+"'"+";"
		# self.env.db.delete_measurement(query)
		self.json_body["tags"]["active"] = False
		self.json_body["fields"]["Host_id"] = -1
		self.destroyAt = self.env.interval
		self.hostid = -1

	def updateUtilizationMetrics(self, data):
		self.ips = data['cpu'] * self.getHost().ipsCap / 100
		self.ram.size = data['memory'] * self.getHost().ramCap.size / 100
		if self.lastReadBytes != 0:
			self.ram.read = (data['read_bytes'] - self.lastReadBytes) / (1024 * 1024 * self.env.intervaltime)
			self.ram.write = (data['write_bytes'] - self.lastWriteBytes) / (1024 * 1024 * self.env.intervaltime)
			self.disk.read = (data['read_bytes'] - self.lastReadBytes) / (1024 * 1024 * self.env.intervaltime)
			self.disk.write = (data['write_bytes'] - self.lastWriteBytes) / (1024 * 1024 * self.env.intervaltime)
		self.lastReadBytes = data['read_bytes']	
		self.lastWriteBytes = data['write_bytes']
		self.disk.size = float(data['disk'][:-1]) if data['disk'][-1] == 'M' else 1024 * float(data['disk'][:-1])
		self.bw.downlink = data['bw_down']
		self.bw.uplink = data['bw_up']
		self.active = data['running']
		if not self.active:
			finished_at = parser.parse(data['finished_at']).replace(tzinfo=None)
			now = datetime.utcnow()
			self.totalExecTime -= abs((now - finished_at).total_seconds())
			self.execError = data['error']