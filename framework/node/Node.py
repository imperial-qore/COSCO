from framework.node.Disk import *
from framework.node.RAM import *
from framework.node.Bandwidth import *
import json
from subprocess import call
from datetime import datetime

class Node():
	# IPS = Million Instructions per second capacity 
	# RAM = Ram in MB capacity
	# Disk = Disk characteristics capacity
	# Bw = Bandwidth characteristics capacity
	def __init__(self, ID, IP,IPS, RAM, Disk, Bw, Powermodel, Environment):
		self.id = ID
		self.ip = IP
		self.ipsCap = IPS
		self.ramCap = RAM
		self.diskCap = Disk
		self.bwCap = Bw
		self.powermodel = Powermodel
		self.powermodel.allocHost(self)
		self.powermodel.host = self
		self.env = Environment
		self.createHost()
		
	def createHost(self):
		host_data= []
		json_body = {
						"measurement":"host",
						"tags": {
									"host_id":self.id,
									"host_ip":self.ip
								},
						"time": datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
						"fields":
								{
									"CPU" :self.ipsCap,
									"RAM_size" :self.ramCap.size,
									"RAM_read" :self.ramCap.read,
									"RAM_write" :str(self.ramCap.write),
									"DISK_size" :self.diskCap.size,
									"DISK_read" :str(self.diskCap.read),
									"DISK_write":str(self.diskCap.write),
									"Bw_up" :self.bwCap.uplink,
									"Bw_down": self.bwCap.downlink,
									"Power" :str(self.powermodel)
								}
					}
			
			
		host_data.append(json_body)
			
		
		#print(host_data)
		#
		#if list(self.db.select("select * from host")):
		#self.db.delete("host")
		self.env.db.insert(host_data)
		#self.env.db.insert(host_data)
		#print(list(self.db.select("select * from host")))



	def getPower(self):
		return self.powermodel.power()

	def getCPU(self):
		# 0 - 100 last interval
		ips = self.getApparentIPS()
		return 100 * (ips / self.ipsCap)

	def getBaseIPS(self):
		# Get base ips count as sum of min ips of all containers
		ips = 0
		containers = self.env.getContainersOfHost(self.id)
		for containerID in containers:
			ips += self.env.getContainerByID(containerID).getBaseIPS()
		# assert ips <= self.ipsCap
		return ips

	def getApparentIPS(self):
		# Give containers remaining IPS for faster execution
		ips = 0
		containers = self.env.getContainersOfHost(self.id)
		for containerID in containers:
			ips += self.env.getContainerByID(containerID).getApparentIPS()
		# assert int(ips) <= self.ipsCap
		return int(ips)

	def getIPSAvailable(self):
		# IPS available is ipsCap - baseIPS
		# When containers allocated, existing ips can be allocated to
		# the containers
		return self.ipsCap - self.getBaseIPS()

	def getCurrentRAM(self):
		size, read, write = 0, 0, 0
		containers = self.env.getContainersOfHost(self.id)
		for containerID in containers:
			s, r, w = self.env.getContainerByID(containerID).getRAM()
			size += s; read += r; write += w
		# assert size <= self.ramCap.size
		# assert read <= self.ramCap.read
		# assert write <= self.ramCap.write
		return size, read, write

	def getRAMAvailable(self):
		size, read, write = self.getCurrentRAM()
		return self.ramCap.size - size, self.ramCap.read - read, self.ramCap.write - write

	def getCurrentDisk(self):
		size, read, write = 0, 0, 0
		containers = self.env.getContainersOfHost(self.id)
		for containerID in containers:
			s, r, w = self.env.getContainerByID(containerID).getDisk()
			size += s; read += r; write += w
		assert size <= self.diskCap.size
		assert read <= self.diskCap.read
		assert write <= self.diskCap.write
		return size, read, write

	def getDiskAvailable(self):
		size, read, write = self.getCurrentDisk()
		return self.diskCap.size - size, self.diskCap.read - read, self.diskCap.write - write
