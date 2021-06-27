from framework.node.Node import *
from framework.server.controller import *
from framework.Framework import *
from time import time, sleep
from pdb import set_trace as bp
import multiprocessing
from joblib import Parallel, delayed
from utils.ColorUtils import *
from workflow.task.Task import *
from pprint import pprint

num_cores = multiprocessing.cpu_count()

class Workflow(Framework):
	# Total power in watt
	# Total Router Bw
	# Interval Time in seconds
	def __init__(self, Scheduler, ContainerLimit, IntervalTime, hostinit, database, env, logger):
		super().__init__(Scheduler, ContainerLimit, IntervalTime, hostinit, database, env, logger)
		self.inactiveContainers = []
		self.destroyedccids = set()
		self.activeworkflows = {}
		self.destroyedworkflows = {}
		self.intervalAllocTimings = []
		self.globalStartTime = time()

	def addContainerInit(self, CreationID, WorkflowID, dependentOn, CreationInterval, SLA, Application, App):
		container = Task(len(self.containerlist), CreationID, WorkflowID, dependentOn, CreationInterval, SLA, Application, App, self, HostID = -1)
		self.containerlist.append(container)
		return container

	def addContainerListInit(self, containerInfoList):
		maxdeploy = min(len(containerInfoList), self.containerlimit-self.getNumActiveContainers())
		deployedContainers = []
		for CreationID, WorkflowID, dependentOn, CreationInterval, SLA, Application, App in containerInfoList:
			if dependentOn is None or set(dependentOn) <= self.destroyedccids:
				dep = self.addContainerInit(CreationID, WorkflowID, dependentOn, CreationInterval, SLA, Application, App)
				deployedContainers.append(dep)
				if len(deployedContainers) >= maxdeploy: break
		self.containerlist += [None] * (self.containerlimit - len(self.containerlist))
		return [container.id for container in deployedContainers]

	def addContainersInit(self, containerInfoListInit):
		self.interval += 1
		self.addWorkflows(containerInfoListInit)
		deployed = self.addContainerListInit(containerInfoListInit)
		return deployed

	def addContainer(self, CreationID, WorkflowID, dependentOn, CreationInterval, SLA, Application, App):
		for i,c in enumerate(self.containerlist):
			if c == None or not c.active:
				container = Task(i, CreationID, WorkflowID, dependentOn, CreationInterval, SLA, Application, App, self, HostID = -1)
				self.containerlist[i] = container
				return container

	def addContainerList(self, containerInfoList):
		maxdeploy = min(len(containerInfoList), self.containerlimit-self.getNumActiveContainers())
		if maxdeploy == 0: return []
		deployedContainers = []
		for CreationID, WorkflowID, dependentOn, CreationInterval, SLA, Application, App in containerInfoList:
			if dependentOn is None or set(dependentOn) <= self.destroyedccids:
				dep = self.addContainer(CreationID, WorkflowID, dependentOn, CreationInterval, SLA, Application, App)
				deployedContainers.append(dep)
				if len(deployedContainers) >= maxdeploy: break
		return [container.id for container in deployedContainers]

	def addContainers(self, containerInfoListInit):
		self.interval += 1
		self.addWorkflows(containerInfoListInit)
		destroyed = self.destroyCompletedContainers()
		deployed = self.addContainerList(containerInfoListInit)
		return deployed, destroyed

	def addWorkflows(self, containerInfoList):
		for CreationID, WorkflowID, _, interval, _, application, _ in containerInfoList:
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
		for (cid, hid) in decision:
			container = self.getContainerByID(cid)
			assert container.getHostID() == -1 and hid != -1
			if self.getPlacementPossible(cid, hid):
				migrations.append((cid, hid))
				container.allocateAndExecute(hid)
			else: 
				self.containerlist[cid] = None
		self.intervalAllocTimings.append(time() - start)
		self.logger.debug("First allocation: "+str(decision))
		self.logger.debug('Interval allocation time for interval '+str(self.interval)+' is '+str(self.intervalAllocTimings[-1]))
		print('Interval allocation time for interval '+str(self.interval)+' is '+str(self.intervalAllocTimings[-1]))
		self.visualSleep(self.intervaltime - self.intervalAllocTimings[-1])
		for host in self.hostlist:
			host.updateUtilizationMetrics()
		return migrations

	def destroyCompletedWorkflows(self):
		toDelete = []
		for WorkflowID in self.activeworkflows:
			allDestroyed = True
			for ccid in self.activeworkflows[WorkflowID]['ccids']:
				if ccid not in self.destroyedccids:
					allDestroyed = False
			if allDestroyed:
				correct, total = self.checkWorkflowOutput(WorkflowID)
				shutil.rmtree('tmp/'+str(WorkflowID)+'/')
				self.destroyedworkflows[WorkflowID] = deepcopy(self.activeworkflows[WorkflowID])
				self.destroyedworkflows[WorkflowID]['sla'] = self.activeworkflows[WorkflowID]['sla']
				self.destroyedworkflows[WorkflowID]['destroyAt'] = self.interval
				self.destroyedworkflows[WorkflowID]['result'] = (correct, total)
				print(color.GREEN); print("Workflow ID: ", WorkflowID)
				pprint(self.destroyedworkflows[WorkflowID]); print(color.ENDC)
				toDelete.append(WorkflowID)
		for WorkflowID in toDelete:
			del self.activeworkflows[WorkflowID]

	def destroyCompletedContainers(self):
		destroyed = []
		for i, container in enumerate(self.containerlist):
			if container and not container.active:
				container.destroy()
				self.destroyedccids.add(container.creationID)
				self.containerlist[i] = None
				self.inactiveContainers.append(container)
				destroyed.append(container)
		self.destroyCompletedWorkflows()
		return destroyed

	def parallelizedFunc(self, i):
		cid, hid = i
		container = self.getContainerByID(cid)
		if self.containerlist[cid].hostid != -1:
			container.allocateAndrestore(hid)
		else:
			container.allocateAndExecute(hid)
		return container

	def visualSleep(self, t):
		total = str(int(t//60))+" min, "+str(t%60)+" sec"
		for i in range(int(t)):
			print("\r>> Interval timer "+str(i//60)+" min, "+str(i%60)+" sec of "+total, end=' ')
			sleep(1)
		sleep(t % 1)
		print()

	def simulationStep(self, decision):
		start = time()
		migrations = []
		containerIDsAllocated = []
		print(decision)
		for (cid, hid) in decision:
			container = self.getContainerByID(cid)
			currentHostID = self.getContainerByID(cid).getHostID()
			currentHost = self.getHostByID(currentHostID)
			targetHost = self.getHostByID(hid)
			if hid != self.containerlist[cid].hostid and self.getPlacementPossible(cid, hid):
				containerIDsAllocated.append(cid)
				migrations.append((cid, hid))
		Parallel(n_jobs=num_cores, backend='threading')(delayed(self.parallelizedFunc)(i) for i in migrations)
		for (cid, hid) in decision:
			if self.containerlist[cid].hostid == -1: self.containerlist[cid] = None
		self.intervalAllocTimings.append(time() - start)
		self.logger.debug("Decision: "+str(decision))
		self.logger.debug('Interval allocation time for interval '+str(self.interval)+' is '+str(self.intervalAllocTimings[-1]))
		print('Interval allocation time for interval '+str(self.interval)+' is '+str(self.intervalAllocTimings[-1]))
		self.visualSleep(max(0, self.intervaltime - self.intervalAllocTimings[-1]))
		for host in self.hostlist:
			host.updateUtilizationMetrics()
		return migrations
