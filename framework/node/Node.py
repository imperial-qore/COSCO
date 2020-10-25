from metrics.Disk import *
from metrics.RAM import *
from metrics.Bandwidth import *
import json
from subprocess import call
from datetime import datetime

class Node():
	# IPS = Million Instructions per second capacity 
	# RAM = Ram in MB capacity
	# Disk = Disk characteristics capacity
	# Bw = Bandwidth characteristics capacity
	def __init__(self, ID, IP, IPS, RAM_, Disk_, Bw, Powermodel, Framework):
		self.id = ID
		self.ip = IP
		self.ipsCap = IPS
		self.ramCap = RAM_
		self.diskCap = Disk_
		self.bwCap = Bw
		# Initialize utilization metrics
		self.ips = 0
		self.ram = RAM(0, 0, 0)
		self.bw = Bandwidth(0, 0)
		self.disk = Disk(0, 0, 0)
		self.json_body = {}
		self.powermodel = Powermodel
		self.powermodel.allocHost(self)
		self.powermodel.host = self
		self.env = Framework
		self.createHost()
		
	def createHost(self):
		self.json_body = {
						"measurement":"host",
						"tags": {
									"host_id":self.id,
									"host_ip":self.ip
								},
						"time": datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
						"interval": self.env.interval,
						"fields": {
									"IPS_Cap": self.ipsCap,
									"RAM_Cap_size": self.ramCap.size,
									"RAM_Cap_read": self.ramCap.read,
									"RAM_Cap_write": self.ramCap.write,
									"DISK_Cap_size": self.diskCap.size,
									"DISK_Cap_read": self.diskCap.read,
									"DISK_Cap_write": self.diskCap.write,
									"Bw_Cap_up": self.bwCap.uplink,
									"Bw_Cap_down": self.bwCap.downlink,
									"IPS": self.ips,
									"RAM_size": self.ram.size,
									"RAM_read": self.ram.read,
									"RAM_write": self.ram.write,
									"DISK_size": self.disk.size,
									"DISK_read": self.disk.read,
									"DISK_write": self.disk.write,
									"Bw_up": self.bw.uplink,
									"Bw_down": self.bw.downlink,
									"Power": str(self.powermodel.__class__.__name__)
								}
					}
		self.env.db.insert([self.json_body])

	def getPower(self):
		return self.powermodel.power()

	def getPowerFromIPS(self, ips):
		return self.powermodel.powerFromCPU(min(100, 100 * (ips / self.ipsCap)))

	def getCPU(self):
		# 0 - 100 last interval
		return min(100, 100 * (self.ips / self.ipsCap))

	def getBaseIPS(self):
		return self.ips

	def getApparentIPS(self):
		return self.ips

	def getIPSAvailable(self):
		return self.ipsCap - self.ips

	def getCurrentRAM(self):
		return self.ram.size, self.ram.read, self.ram.write

	def getRAMAvailable(self):
		size, read, write = self.getCurrentRAM()
		return max(0, (0.6 if self.ramCap.size < 4000 else 0.8) * self.ramCap.size - size), self.ramCap.read - read, self.ramCap.write - write

	def getCurrentDisk(self):
		return self.disk.size, self.disk.read, self.disk.write

	def getDiskAvailable(self):
		size, read, write = self.getCurrentDisk()
		return self.diskCap.size - size, self.diskCap.read - read, self.diskCap.write - write

	def updateUtilizationMetrics(self):
		container_data, _ = self.env.controller.getContainerStat(self.ip)
		for container_d in container_data:
			ccid = int(container_d['fields']['name'].split("_")[0])
			container = self.env.getContainerByCID(ccid)
			container.updateUtilizationMetrics(container_d['fields'])
		host_data, _ = self.env.controller.gethostStat(self.ip)
		if 'fields' in host_data:
			self.ips = host_data['fields']['cpu'] * self.ipsCap / 100
			self.ram.size = host_data['fields']['memory']
			self.disk.size = host_data['fields']['disk']
		self.ram.read, self.ram.write = 0, 0
		self.disk.read, self.disk.write = 0, 0
		for cid in self.env.getContainersOfHost(self.id):
			self.ram.read += self.env.containerlist[cid].ram.read
			self.disk.read += self.env.containerlist[cid].disk.read
			self.ram.write += self.env.containerlist[cid].ram.write
			self.disk.write += self.env.containerlist[cid].ram.write