import numpy as np
from simulator.host.Disk import *
from simulator.host.RAM import *
from simulator.host.Bandwidth import *
from metrics.powermodels.PMRaspberryPi import *
from metrics.powermodels.PMB2s import *
from metrics.powermodels.PMB4ms import *
from metrics.powermodels.PMB8ms import *
from metrics.powermodels.PMXeon_X5570 import *

class AzureFog():
	def __init__(self, num_hosts):
		self.num_hosts = num_hosts
		self.types = {
			'B2s':
				{
					'IPS': 4029,
					'RAMSize': 4295,
					'RAMRead': 372.0,
					'RAMWrite': 200.0,
					'DiskSize': 32212,
					'DiskRead': 13.42,
					'DiskWrite': 1.011,
					'BwUp': 5000,
					'BwDown': 5000,
					'Power': 'PMB2s'
				},
			'B4ms':
				{
					'IPS': 4029,
					'RAMSize': 17180,
					'RAMRead': 360.0,
					'RAMWrite': 305.0,
					'DiskSize': 32212,
					'DiskRead': 10.38,
					'DiskWrite': 0.619,
					'BwUp': 5000,
					'BwDown': 5000,
					'Power': 'PMB4ms'
				},
			'B8ms':
				{
					'IPS': 16111,
					'RAMSize': 34360,
					'RAMRead': 376.54,
					'RAMWrite': 266.75,
					'DiskSize': 32212,
					'DiskRead': 11.64,
					'DiskWrite': 1.164,
					'BwUp': 5000,
					'BwDown': 5000,
					'Power': 'PMB8ms'
				}
 		}

	def generateHosts(self):
		hosts = []
		types = ['B2s', 'B2s', 'B2s', 'B2s', 'B4ms', 'B4ms', 'B4ms', 'B4ms', 'B8ms', 'B8ms']
		for i in range(self.num_hosts):
			typeID = types[i]
			IPS = self.types[typeID]['IPS']
			Ram = RAM(self.types[typeID]['RAMSize'], self.types[typeID]['RAMRead']*5, self.types[typeID]['RAMWrite']*5)
			Disk_ = Disk(self.types[typeID]['DiskSize'], self.types[typeID]['DiskRead']*5, self.types[typeID]['DiskWrite']*10)
			Bw = Bandwidth(self.types[typeID]['BwUp'], self.types[typeID]['BwDown'])
			Power = eval(self.types[typeID]['Power']+'()')
			hosts.append((IPS, Ram, Disk_, Bw, Power))
		return hosts