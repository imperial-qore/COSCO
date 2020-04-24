import numpy as np
import matplotlib.pyplot as plt
plt.style.use(['science'])
plt.rcParams["text.usetex"] = False

class Stats():
	def __init__(self, Environment, WorkloadModel, Datacenter, Scheduler):
		self.env = Environment
		self.workload = WorkloadModel
		self.datacenter = Datacenter
		self.scheduler = Scheduler
		self.initStats()

	def initStats(self):	
		self.hostinfo = []
		self.workloadinfo = []
		self.activecontainerinfo = []
		self.allcontainerinfo = []
		self.metrics = []
		self.schedulerinfo = []

	def saveHostInfo(self):
		hostinfo = dict()
		hostinfo['interval'] = self.env.interval
		hostinfo['cpu'] = [host.getCPU() for host in self.env.hostlist]
		hostinfo['numcontainers'] = [len(self.env.getContainersOfHost(i)) for i,host in enumerate(self.env.hostlist)]
		hostinfo['power'] = [host.getPower() for host in self.env.hostlist]
		hostinfo['baseips'] = [host.getBaseIPS() for host in self.env.hostlist]
		hostinfo['ipsavailable'] = [host.getIPSAvailable() for host in self.env.hostlist]
		hostinfo['apparentips'] = [host.getApparentIPS() for host in self.env.hostlist]
		hostinfo['ram'] = [host.getCurrentRAM() for host in self.env.hostlist]
		hostinfo['ramavailable'] = [host.getRAMAvailable() for host in self.env.hostlist]
		hostinfo['disk'] = [host.getCurrentDisk() for host in self.env.hostlist]
		hostinfo['diskavailable'] = [host.getDiskAvailable() for host in self.env.hostlist]
		self.hostinfo.append(hostinfo)

	def saveWorkloadInfo(self, deployed, migrations):
		workloadinfo = dict()
		workloadinfo['interval'] = self.env.interval
		workloadinfo['totalcontainers'] = len(self.workload.createdContainers)
		if self.workloadinfo != []:
			workloadinfo['newcontainers'] = workloadinfo['totalcontainers'] - self.workloadinfo[-1]['totalcontainers'] 
		else:
			workloadinfo['newcontainers'] = workloadinfo['totalcontainers']
		workloadinfo['deployed'] = len(deployed)
		workloadinfo['migrations'] = len(migrations)
		workloadinfo['inqueue'] = len(self.workload.getUndeployedContainers())
		self.workloadinfo.append(workloadinfo)

	def saveContainerInfo(self):
		containerinfo = dict()
		containerinfo['interval'] = self.env.interval
		containerinfo['activecontainers'] = self.env.getNumActiveContainers()
		containerinfo['ips'] = [(c.getBaseIPS() if c else 0) for c in self.env.containerlist]
		containerinfo['apparentips'] = [(c.getApparentIPS() if c else 0) for c in self.env.containerlist]
		containerinfo['ram'] = [(c.getRAM() if c else 0) for c in self.env.containerlist]
		containerinfo['disk'] = [(c.getDisk() if c else 0) for c in self.env.containerlist]
		containerinfo['creationids'] = [(c.creationID if c else -1) for c in self.env.containerlist]
		containerinfo['hostalloc'] = [(c.getHostID() if c else -1) for c in self.env.containerlist]
		containerinfo['active'] = [(c.active if c else False) for c in self.env.containerlist]
		self.activecontainerinfo.append(containerinfo)

	def saveAllContainerInfo(self):
		containerinfo = dict()
		allCreatedContainers = [self.env.getContainerByCID(cid) for cid in list(np.where(self.workload.deployedContainers)[0])]
		containerinfo['interval'] = self.env.interval
		containerinfo['ips'] = [(c.getBaseIPS() if c.active else 0) for c in allCreatedContainers]
		containerinfo['create'] = [(c.createAt) for c in allCreatedContainers]
		containerinfo['start'] = [(c.startAt) for c in allCreatedContainers]
		containerinfo['destroy'] = [(c.destroyAt) for c in allCreatedContainers]
		containerinfo['apparentips'] = [(c.getApparentIPS() if c.active else 0) for c in allCreatedContainers]
		containerinfo['ram'] = [(c.getRAM() if c.active else 0) for c in allCreatedContainers]
		containerinfo['disk'] = [(c.getDisk() if c.active else 0) for c in allCreatedContainers]
		containerinfo['hostalloc'] = [(c.getHostID() if c.active else -1) for c in allCreatedContainers]
		containerinfo['active'] = [(c.active) for c in allCreatedContainers]
		self.activecontainerinfo.append(containerinfo)

	def saveMetrics(self, destroyed, migrations):
		metrics = dict()
		metrics['interval'] = self.env.interval
		metrics['numdestroyed'] = len(destroyed)
		metrics['nummigrations'] = len(migrations)
		metrics['energy'] = [host.getPower()*self.env.intervaltime for host in self.env.hostlist]
		metrics['energytotalinterval'] = np.sum(metrics['energy'])
		metrics['responsetime'] = [c.totalExecTime + c.totalMigrationTime for c in destroyed]
		metrics['avgresponsetime'] = np.average(metrics['responsetime']) if len(destroyed) > 0 else 0
		metrics['migrationtime'] = [c.totalMigrationTime for c in destroyed]
		metrics['avgmigrationtime'] = np.average(metrics['migrationtime']) if len(destroyed) > 0 else 0
		metrics['slaviolations'] = len(np.where([c.destroyAt > c.ipsmodel.SLA for c in destroyed]))
		metrics['slaviolationspercentage'] = metrics['slaviolations'] * 100.0 / len(destroyed) if len(destroyed) > 0 else 0
		metrics['waittime'] = [c.startAt - c.createAt for c in destroyed]
		self.metrics.append(metrics)

	def saveSchedulerInfo(self, selectedcontainers, decision):
		schedulerinfo = dict()
		schedulerinfo['interval'] = self.env.interval
		schedulerinfo['selection'] = selectedcontainers
		schedulerinfo['decision'] = decision
		schedulerinfo['schedule'] = [(c.id, c.getHostID()) if c else (None, None) for c in self.env.containerlist]
		self.schedulerinfo.append(schedulerinfo)

	def saveStats(self, deployed, migrations, destroyed, selectedcontainers, decision):	
		self.saveHostInfo()
		self.saveWorkloadInfo(deployed, migrations)
		self.saveContainerInfo()
		self.saveAllContainerInfo()
		self.saveMetrics(destroyed, migrations)
		self.saveSchedulerInfo(selectedcontainers, decision)

	########################################################################################################

	def generateGraphsWithInterval(self, dirname, listinfo, metric, metric2=None):
		fig, axes = plt.subplots(len(listinfo[0][metric]), 1, sharex=True,figsize=(4, 5))
		title = 'host_' + metric + '_with_interval' 
		totalIntervals = len(listinfo)
		x = list(range(totalIntervals))
		metric_with_interval = []; metric2_with_interval = []
		ylimit = 0; ylimit2 = 0
		for hostID in range(len(listinfo[0][metric])):
			metric_with_interval.append([listinfo[interval][metric][hostID] for interval in range(totalIntervals)])
			ylimit = max(ylimit, max(metric_with_interval[-1]))
			if metric2:
				metric2_with_interval.append([listinfo[interval][metric2][hostID] for interval in range(totalIntervals)])
				ylimit2 = max(ylimit, max(metric2_with_interval[-1]))
		for hostID in range(len(listinfo[0][metric])):
			axes[hostID].set_ylim(0, max(ylimit, ylimit2))
			axes[hostID].plot(x, metric_with_interval[hostID])
			if metric2:
				axes[hostID].plot(x, metric2_with_interval[hostID])
			axes[hostID].set_ylabel('Host '+str(hostID))
		plt.tight_layout(pad=0)
		plt.savefig(dirname + '/' + title + '.pdf')

	def generateGraphs(self, dirname):
		self.generateGraphsWithInterval(dirname, self.hostinfo, 'cpu')
		self.generateGraphsWithInterval(dirname, self.hostinfo, 'numcontainers')
		self.generateGraphsWithInterval(dirname, self.hostinfo, 'power')
		self.generateGraphsWithInterval(dirname, self.hostinfo, 'baseips', 'apparentips')