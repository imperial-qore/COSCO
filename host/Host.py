from .Disk import *
from .RAM import *
from .Bw import *

class Host():
	# IPS = Instructions per second capacity
	# RAM = Ram in MB capacity
	# Disk = Disk characteristics capacity
	# Bw = Bandwidth characteristics capacity
	def __init__(self, ID, IPS, RAM, Disk, Bw, Powermodel, Environment):
		self.id = ID
		self.ipsCap = IPS
		self.ramCap = RAM
		self.diskCap = Disk
		self.bwCap = Bw
		self.powermodel = Powermodel
		self.powermodel.host = self
		self.env = Environment

	def getPower(self):
		return self.powermodel.power(self.getCPU())

	def getCPU(self):
		ips = self.getCurrentIPS()
		return 100 * (ips / self.ipsCap)

	def getCurrentIPS(self):
		ips = 0
		containers = self.env.getContainersOfHost(self.id)
		for containerID in containers:
			ips += self.env.getContainerByID(containerID).getIPS()
		assert ips <= self.ipsCap
		return ips

	def getIPSAvailable(self):
		return self.ipsCap - self.getCurrentIPS()

	def getCurrentRAM(self):
		size, read, write = 0, 0, 0
		containers = self.env.getContainersOfHost(self.id)
		for containerID in containers:
			s, r, w = self.env.getContainerByID(containerID).getRAM()
			size += s; read += r; write += w
		assert size <= self.ramCap.size
		assert read <= self.ramCap.read
		assert write <= self.ramCap.write
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
