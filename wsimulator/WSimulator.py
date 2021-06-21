from simulator.Simulator import *
from simulator.host.Host import *
from wsimulator.container.Container import *
from time import time
from utils.ColorUtils import *
from pprint import pprint
from copy import deepcopy

class WSimulator(Simulator):
	def __init__(self, TotalPower, RouterBw, Scheduler, ContainerLimit, IntervalTime, hostinit):
		super().__init__(TotalPower, RouterBw, Scheduler, ContainerLimit, IntervalTime, hostinit)
		self.inactiveContainers = []
		self.destroyedccids = set()
		self.activeworkflows = {}
		self.destroyedworkflows = {}
		self.intervalAllocTimings = []
		self.globalStartTime = time()

	def addContainerInit(self, CreationID, WorkflowID, dependentOn, CreationInterval, IPSModel, RAMModel, DiskModel, application):
		container = Container(len(self.containerlist), CreationID, WorkflowID, dependentOn, CreationInterval, IPSModel, RAMModel, DiskModel, application, self, HostID = -1)
		self.containerlist.append(container)
		return container

	def addContainerListInit(self, containerInfoList):
		maxdeploy = min(len(containerInfoList), self.containerlimit-self.getNumActiveContainers())
		deployedContainers = []
		for CreationID, WorkflowID, dependentOn, CreationInterval, IPSModel, RAMModel, DiskModel, application in containerInfoList:
			if dependentOn is None or set(dependentOn) <= self.destroyedccids:
				dep = self.addContainerInit(CreationID, WorkflowID, dependentOn, CreationInterval, IPSModel, RAMModel, DiskModel, application)
				deployedContainers.append(dep)
				if len(deployedContainers) >= maxdeploy: break
		self.containerlist += [None] * (self.containerlimit - len(self.containerlist))
		return [container.id for container in deployedContainers]

	def addContainer(self, CreationID, WorkflowID, dependentOn, CreationInterval, IPSModel, RAMModel, DiskModel, application):
		for i,c in enumerate(self.containerlist):
			if c == None or not c.active:
				container = Container(i, CreationID, WorkflowID, dependentOn, CreationInterval, IPSModel, RAMModel, DiskModel, application, self, HostID = -1)
				self.containerlist[i] = container
				return container

	def addContainerList(self, containerInfoList):
		maxdeploy = min(len(containerInfoList), self.containerlimit-self.getNumActiveContainers())
		if maxdeploy == 0: return []
		deployedContainers = []
		for CreationID, WorkflowID, dependentOn, CreationInterval, IPSModel, RAMModel, DiskModel, application in containerInfoList:
			if dependentOn is None or set(dependentOn) <= self.destroyedccids:
				dep = self.addContainer(CreationID, WorkflowID, dependentOn, CreationInterval, IPSModel, RAMModel, DiskModel, application)
				deployedContainers.append(dep)
				if len(deployedContainers) >= maxdeploy: break
		return [container.id for container in deployedContainers]

	def addWorkflows(self, containerInfoList):
		for CreationID, WorkflowID, _, interval, _, _, _, application in containerInfoList:
			if WorkflowID not in self.activeworkflows:
				self.activeworkflows[WorkflowID] = {'ccids': [CreationID], \
					'createAt': interval, \
					'startAt': -1, \
					'application': application}
			elif CreationID not in self.activeworkflows[WorkflowID]['ccids']:
				self.activeworkflows[WorkflowID]['ccids'].append(CreationID)
		print(color.YELLOW); pprint(self.activeworkflows); print(color.ENDC)

	def allocateInit(self, decision):
		start = time()
		migrations = []
		routerBwToEach = self.totalbw / len(decision)
		for (cid, hid) in decision:
			container = self.getContainerByID(cid)
			assert container.getHostID() == -1 and hid != -1
			numberAllocToHost = len(self.scheduler.getMigrationToHost(hid, decision))
			allocbw = min(self.getHostByID(hid).bwCap.downlink / numberAllocToHost, routerBwToEach)
			if self.getPlacementPossible(cid, hid):
				if container.getHostID() != hid:
					migrations.append((cid, hid))
				container.allocateAndExecute(hid, allocbw)
				if self.activeworkflows[container.workflowID]['startAt'] == -1:
					self.activeworkflows[container.workflowID]['startAt'] = self.interval
			else: 
				self.containerlist[cid] = None
		self.intervalAllocTimings.append(time() - start)
		print('Interval allocation time for interval '+str(self.interval)+' is '+str(self.intervalAllocTimings[-1]))
		return migrations

	def addContainersInit(self, containerInfoListInit):
		self.interval += 1
		self.addWorkflows(containerInfoListInit)
		deployed = self.addContainerListInit(containerInfoListInit)
		return deployed

	def addContainers(self, containerInfoListInit):
		self.interval += 1
		self.addWorkflows(containerInfoListInit)
		destroyed = self.destroyCompletedContainers()
		deployed = self.addContainerList(containerInfoListInit)
		return deployed, destroyed

	def destroyCompletedWorkflows(self):
		toDelete = []
		for WorkflowID in self.activeworkflows:
			allDestroyed = True
			for ccid in self.activeworkflows[WorkflowID]['ccids']:
				if ccid not in self.destroyedccids:
					allDestroyed = False
			if allDestroyed:
				self.destroyedworkflows[WorkflowID] = deepcopy(self.activeworkflows[WorkflowID])
				self.destroyedworkflows[WorkflowID]['destroyAt'] = self.interval
				print(color.GREEN); print("Workflow ID: ", WorkflowID)
				pprint(self.destroyedworkflows[WorkflowID]); print(color.ENDC)
				toDelete.append(WorkflowID)
		for WorkflowID in toDelete:
			del self.activeworkflows[WorkflowID]

	def destroyCompletedContainers(self):
		destroyed = []
		for i,container in enumerate(self.containerlist):
			if container and container.getBaseIPS() == 0:
				container.destroy()
				self.destroyedccids.add(container.creationID)
				self.containerlist[i] = None
				self.inactiveContainers.append(container)
				destroyed.append(container)
		self.destroyCompletedWorkflows()
		return destroyed

	def simulationStep(self, decision):
		start = time()
		routerBwToEach = self.totalbw / len(decision) if len(decision) > 0 else self.totalbw
		migrations = []
		containerIDsAllocated = []
		for (cid, hid) in decision:
			container = self.getContainerByID(cid)
			currentHostID = self.getContainerByID(cid).getHostID()
			currentHost = self.getHostByID(currentHostID)
			targetHost = self.getHostByID(hid)
			migrateFromNum = len(self.scheduler.getMigrationFromHost(currentHostID, decision))
			migrateToNum = len(self.scheduler.getMigrationToHost(hid, decision))
			allocbw = min(targetHost.bwCap.downlink / migrateToNum, currentHost.bwCap.uplink / migrateFromNum, routerBwToEach)
			if hid != self.containerlist[cid].hostid and self.getPlacementPossible(cid, hid):
				migrations.append((cid, hid))
				container.allocateAndExecute(hid, allocbw)
				containerIDsAllocated.append(cid)
				if self.activeworkflows[container.workflowID]['startAt'] == -1:
					self.activeworkflows[container.workflowID]['startAt'] = self.interval
		# destroy pointer to unallocated containers as book-keeping is done by workload model
		for (cid, hid) in decision:
			if self.containerlist[cid].hostid == -1: self.containerlist[cid] = None
		self.intervalAllocTimings.append(time() - start)
		print('Interval allocation time for interval '+str(self.interval)+' is '+str(self.intervalAllocTimings[-1]))
		for i,container in enumerate(self.containerlist):
			if container and i not in containerIDsAllocated:
				container.execute(0)
		return migrations