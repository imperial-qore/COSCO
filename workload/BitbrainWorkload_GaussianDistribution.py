from .Workload import *
from container.IPSModels.IPSMBitbrain import *
from container.RAMModels.RMBitbrain import *
from container.DiskModels.DMBitbrain import *
from random import gauss, randint
from os import path, makedirs, listdir, remove
import wget
from zipfile import ZipFile
import shutil
import pandas as pd
import warnings
warnings.simplefilter("ignore")

# Intel Pentium III gives 2054 MIPS at 600 MHz
# Source: https://archive.vn/20130205075133/http://www.tomshardware.com/charts/cpu-charts-2004/Sandra-CPU-Dhrystone,449.html
ips_multiplier = 2054 / 600 

class BWGD(Workload):
	def __init__(self, meanNumContainers, sigmaNumContainers):
		super().__init__()
		self.mean = meanNumContainers
		self.sigma = sigmaNumContainers
		dataset_path = 'workload/datasets/bitbrain/'
		if not path.exists(dataset_path):
			makedirs(dataset_path)
			print('Downloading Bitbrain Dataset')
			url = 'http://gwa.ewi.tudelft.nl/fileadmin/pds/trace-archives/grid-workloads-archive/datasets/gwa-t-12/rnd.zip'
			filename = wget.download(url); zf = ZipFile(filename, 'r'); zf.extractall(dataset_path); zf.close()
			for f in listdir(dataset_path+'rnd/2013-9/'): shutil.move(dataset_path+'rnd/2013-9/'+f, dataset_path+'rnd/')
			shutil.rmtree(dataset_path+'rnd/2013-7'); shutil.rmtree(dataset_path+'rnd/2013-8')
			shutil.rmtree(dataset_path+'rnd/2013-9'); remove(filename)
		self.dataset_path = dataset_path
		self.disk_sizes = [100, 200, 300, 400, 500]
		self.meanSLA, self.sigmaSLA = 20, 3

	def generateNewContainers(self, interval):
		workloadlist = []
		for i in range(max(1,int(gauss(self.mean, self.sigma)))):
			CreationID = self.creation_id
			index = randint(1,500)
			df = pd.read_csv(self.dataset_path+'rnd/'+str(index)+'.csv', sep=';\t')
			sla = gauss(self.meanSLA, self.sigmaSLA)
			IPSModel = IPSMBitbrain((ips_multiplier*df['CPU usage [MHZ]']).to_list(), (ips_multiplier*df['CPU capacity provisioned [MHZ]']).to_list()[0], int(1.2*sla), interval + sla)
			RAMModel = RMBitbrain((df['Memory capacity provisioned [KB]']/1000).to_list(), (df['Network received throughput [KB/s]']/1000).to_list(), (df['Network transmitted throughput [KB/s]']/1000).to_list())
			disk_size  = self.disk_sizes[index % len(self.disk_sizes)]
			DiskModel = DMBitbrain(disk_size, (df['Disk read throughput [KB/s]']/1000).to_list(), (df['Disk write throughput [KB/s]']/1000).to_list())
			workloadlist.append((CreationID, interval, IPSModel, RAMModel, DiskModel))
			self.creation_id += 1
		self.createdContainers += workloadlist
		self.deployedContainers += [False] * len(workloadlist)
		return self.getUndeployedContainers()