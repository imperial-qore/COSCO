import numpy as np
from host.Disk import *
from host.RAM import *
from host.Bandwidth import *
from host.Powermodels.PMRaspberryPi import *
from host.Powermodels.PMConstant import *

class SimpleFog():
	def __init__(self, num_hosts):
		self.num_hosts = num_hosts
		self.types = {
			'IPS' : [20000, 30000, 40000],
			'RAMSize' : [3000, 4000, 8000],
			'RAMRead' : [3000, 2000, 3000],
			'RAMWrite' : [3000, 2000, 3000],
			'DiskSize' : [30000, 40000, 80000],
			'DiskRead' : [2000, 2000, 3000],
			'DiskWrite' : [2000, 2000, 3000],
			'BwUp' : [1000, 2000, 5000],
			'BwDown': [2000, 4000, 10000],
			'Power' : [1]
 		}

	def generateHosts(self):
		hosts = []
		for i in range(self.num_hosts):
			typeID = np.random.randint(0,3) # i%3 #
			IPS = self.types['IPS'][typeID]
			Ram = RAM(self.types['RAMSize'][typeID], self.types['RAMRead'][typeID], self.types['RAMWrite'][typeID])
			Disk_ = Disk(self.types['DiskSize'][typeID], self.types['DiskRead'][typeID], self.types['DiskWrite'][typeID])
			Bw = Bandwidth(self.types['BwUp'][typeID], self.types['BwDown'][typeID])
			Power = PMConstant(self.types['Power'][typeID]) if typeID < 1 else PMRaspberryPi()
			hosts.append((IPS, Ram, Disk_, Bw, Power))
		return hosts