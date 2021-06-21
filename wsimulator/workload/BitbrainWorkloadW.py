from .Workload import *
from simulator.container.IPSModels.IPSMBitbrain import *
from simulator.container.RAMModels.RMBitbrain import *
from simulator.container.DiskModels.DMBitbrain import *
from random import gauss, randint, choices
from os import path, makedirs, listdir, remove
import wget
import shutil
import pandas as pd
import warnings
warnings.simplefilter("ignore")

# Intel Pentium III gives 2054 MIPS at 600 MHz
# Source: https://archive.vn/20130205075133/http://www.tomshardware.com/charts/cpu-charts-2004/Sandra-CPU-Dhrystone,449.html
ips_multiplier = 2054.0 / (2 * 600)

class BWGD2W(Workload):
	def __init__(self, meanNumContainers, sigmaNumContainers):
		super().__init__()
		self.mean = meanNumContainers
		self.sigma = sigmaNumContainers
		self.workflows = ['Type1', 'Type2', 'Type3']
		dataset_path = 'simulator/workload/datasets/bitbrain/'
		if not path.exists(dataset_path):
			makedirs(dataset_path)
			print('Downloading Bitbrain Dataset')
			url = 'http://gwa.ewi.tudelft.nl/fileadmin/pds/trace-archives/grid-workloads-archive/datasets/gwa-t-12/rnd.zip'
			filename = wget.download(url); zf = ZipFile(filename, 'r'); zf.extractall(dataset_path); zf.close()
			for f in listdir(dataset_path+'rnd/2013-9/'): shutil.move(dataset_path+'rnd/2013-9/'+f, dataset_path+'rnd/')
			shutil.rmtree(dataset_path+'rnd/2013-7'); shutil.rmtree(dataset_path+'rnd/2013-8')
			shutil.rmtree(dataset_path+'rnd/2013-9'); remove(filename)
		self.dataset_path = dataset_path
		self.disk_sizes = [1, 2, 3]
		self.minSLA, self.meanSLA, self.sigmaSLA = 2, 10, 3
		self.possible_indices = []
		for i in range(1, 500):
			df = pd.read_csv(self.dataset_path+'rnd/'+str(i)+'.csv', sep=';\t')
			if (ips_multiplier*df['CPU usage [MHZ]']).to_list()[10] < 3000 and (ips_multiplier*df['CPU usage [MHZ]']).to_list()[10] > 500:
				self.possible_indices.append(i)			

	def generateNewWorkflows(self, interval):
		workflowlist = []
		minimum_workflows = 1 if interval == 0 else 0
		for i in range(max(minimum_workflows,int(gauss(self.mean, self.sigma)))):
			WorkflowID = self.workflow_id
			workflow = choices(self.workflows, weights=[0.5, 0.25, 0.25])[0]
			SLA = max(self.minSLA, gauss(self.meanSLA, self.sigmaSLA))
			workflowlist.append((WorkflowID, interval, SLA, workflow))
			self.workflow_id += 1
		return workflowlist

	def generateRandomContainer(self, WorkflowID, dependentOn, interval, application):
		CreationID = self.creation_id
		index = self.possible_indices[randint(0,len(self.possible_indices)-1)]
		df = pd.read_csv(self.dataset_path+'rnd/'+str(index)+'.csv', sep=';\t')
		sla = gauss(self.meanSLA, self.sigmaSLA)
		IPSModel = IPSMBitbrain((ips_multiplier*df['CPU usage [MHZ]']).to_list(), (ips_multiplier*df['CPU capacity provisioned [MHZ]']).to_list()[0], int(1.2*sla), interval + sla)
		RAMModel = RMBitbrain((df['Memory usage [KB]']/4000).to_list(), (df['Network received throughput [KB/s]']/1000).to_list(), (df['Network transmitted throughput [KB/s]']/1000).to_list())
		disk_size  = self.disk_sizes[index % len(self.disk_sizes)]
		DiskModel = DMBitbrain(disk_size, (df['Disk read throughput [KB/s]']/4000).to_list(), (df['Disk write throughput [KB/s]']/12000).to_list())
		self.creation_id += 1
		return CreationID, (WorkflowID, dependentOn, interval, IPSModel, RAMModel, DiskModel, application)

	def generateNewContainers(self, interval):
		workloadlist = []
		workflowlist = self.generateNewWorkflows(interval)
		for i, (WorkflowID, interval, SLA, workflow) in enumerate(workflowlist):
			if workflow == 'Type1': 
				# + -> + -> + -> +
				CreationID1, info = self.generateRandomContainer(WorkflowID, None, interval, workflow)
				workloadlist.append((CreationID1, *info))
				CreationID2, info = self.generateRandomContainer(WorkflowID, [CreationID1], interval, workflow)
				workloadlist.append((CreationID2, *info))
				CreationID3, info = self.generateRandomContainer(WorkflowID, [CreationID2], interval, workflow)
				workloadlist.append((CreationID3, *info))
				CreationID4, info = self.generateRandomContainer(WorkflowID, [CreationID3], interval, workflow)
				workloadlist.append((CreationID4, *info))
			if workflow == 'Type2': 
				#      +
				# + ->   -> +
				#      + 
				CreationID1, info = self.generateRandomContainer(WorkflowID, None, interval, workflow)
				workloadlist.append((CreationID1, *info))
				CreationID2, info = self.generateRandomContainer(WorkflowID, [CreationID1], interval, workflow)
				workloadlist.append((CreationID2, *info))
				CreationID3, info = self.generateRandomContainer(WorkflowID, [CreationID1], interval, workflow)
				workloadlist.append((CreationID3, *info))
				CreationID4, info = self.generateRandomContainer(WorkflowID, [CreationID2, CreationID3], interval, workflow)
				workloadlist.append((CreationID4, *info))
			if workflow == 'Type3': 
				#      + -> +
				# + ->   
				#      + -> +
				CreationID1, info = self.generateRandomContainer(WorkflowID, None, interval, workflow)
				workloadlist.append((CreationID1, *info))
				CreationID2, info = self.generateRandomContainer(WorkflowID, [CreationID1], interval, workflow)
				workloadlist.append((CreationID2, *info))
				CreationID3, info = self.generateRandomContainer(WorkflowID, [CreationID1], interval, workflow)
				workloadlist.append((CreationID3, *info))
				CreationID4, info = self.generateRandomContainer(WorkflowID, [CreationID2], interval, workflow)
				workloadlist.append((CreationID4, *info))
				CreationID5, info = self.generateRandomContainer(WorkflowID, [CreationID3], interval, workflow)
				workloadlist.append((CreationID5, *info))
		self.createdContainers += workloadlist
		self.deployedContainers += [False] * len(workloadlist)
		return self.getUndeployedContainers()