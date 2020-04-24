
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
		hostinfo['power'] = [host.getPower() for hsot in self.env.hostlist]
		hostinfo['baseips'] = [host.getBaseIPS() for host in self.env.hostlist]
		hostinfo['ipsavailable'] = [host.getIPSAvailable() for hsot in self.env.hostlist]
		hostinfo['apparentips'] = [host.getApparentIPS() for host in self.env.hostlist]
		hostinfo['ram'] = [host.getRAM() for hsot in self.env.hostlist]
		hostinfo['ramavailable'] = [host.getRAMAvailable() for host in self.env.hostlist]
		hostinfo['disk'] = [host.getDisk() for hsot in self.env.hostlist]
		hostinfo['diskavailable'] = [host.getDiskAvailable() for host in self.env.hostlist]
		self.hostinfo.append(hostinfo)

	def saveWorkloadInfo(self, deployed, migrated, newcontainers):
		workloadinfo = dict()
		workloadinfo['interval'] = self.env.interval
		workloadinfo['totalcontainers'] = len(self.workload.createdContainers)
		if self.workloadinfo != []:
			workloadinfo['newcontainers'] = workloadinfo['totalcontainers'] - self.workloadinfo[-1]['totalcontainers'] 
		else:
			workloadinfo['newcontainers'] = workloadinfo['totalcontainers']
		workloadinfo['new'] = len(newcontainers)
		workloadinfo['deployed'] = len(deployed)
		workloadinfo['migrated'] = len(migrated)
		workloadinfo['inqueue'] = len(self.workload.getUndeployedContainers())
		self.workloadinfo.append(workloadinfo)

	def saveContainerInfo(self):
		containerinfo = dict()
		containerinfo['interval'] = self.env.interval
		containerinfo['activecontainers'] = self.env.getNumActiveContainers()
		containerinfo['ips'] = [(c.getIPS() if c else 0) for c in self.env.containerlist]
		containerinfo['apparentips'] = [(c.getApparentIPS() if c else 0) for c in self.env.containerlist]
		containerinfo['ram'] = [(c.getRAM() if c else 0) for c in self.env.containerlist]
		containerinfo['disk'] = [(c.getDisk() if c else 0) for c in self.env.containerlist]
		containerlist['creationids'] = [(c.creationID if c else -1) for c in self.env.containerlist]
		containerlist['hostalloc'] = [(c.getHostID() if c else -1) for c in self.env.containerlist]
		containerlist['active'] = [(c.active if c else False) for c in self.env.containerlist]
		self.activecontainerinfo.append(containerinfo)

	def saveAllContainerInfo(self):
		containerinfo = dict()
		allContainers = [self.env.getContainerByCID(cid) for cid in range(self.workload.creation_id)]
		containerinfo['interval'] = self.env.interval
		containerinfo['ips'] = [(c.getIPS() if c.active else 0) for c in allContainers]
		containerinfo['create'] = [(c.createAt) for c in allContainers]
		containerinfo['start'] = [(c.startAt) for c in allContainers]
		containerinfo['destroy'] = [(c.destroyAt) for c in allContainers]
		containerinfo['apparentips'] = [(c.getApparentIPS() if c.active else 0) for c in allContainers]
		containerinfo['ram'] = [(c.getRAM() if c.active else 0) for c in self.env.containerlist]
		containerinfo['disk'] = [(c.getDisk() if c.active else 0) for c in self.env.containerlist]
		containerlist['hostalloc'] = [(c.getHostID() if c.active else -1) for c in self.env.containerlist]
		containerlist['active'] = [(c.active) for c in self.env.containerlist]
		self.activecontainerinfo.append(containerinfo)

	def saveMetrics(self, destroyed, migrated):
		metrtics = dict()
		metrics['interval'] = self.env.interval
		metrics['numdestroyed'] = len(destroyed)
		metrics['nummigrations'] = len('migrations')
		metrics['energy'] = [host.getPower()*self.env.intervaltime for hsot in self.env.hostlist]
		metrics['energytotalinterval'] = np.sum(metrics['energy'])
		metrics['responsetime'] = [c.totalExecutionTime + c.totalMigrationTime for c in destroyed]
		metrics['avgresponsetime'] = np.average(metrics['responsetime'])
		metrics['migrationtime'] = [c.totalMigrationTime for c in destroyed]
		metrics['avgmigrationtime'] = np.average(metrics['migrationtime'])
		metrics['slaviolations'] = len(np.where([c.destroyAt > c.SLA for c in destroyed]))
		metrics['slaviolationspercentage'] = metrics['slaviolations'] * 100.0 / len(destroyed)
		metrics['waittime'] = [c.startAt - c.createAt for c in destroyed]
		self.metrics.append(metrics)

	def saveSchedulerInfo(self):
		schedulerinfo = dict()
		schedulerinfo['interval'] = self.env.interval
		schedulerinfo['selection'] = selectedcontainers
		schedulerinfo['decision'] = decision
		schedulerinfo['schedule'] = [(c.containerID, c.getHostID()) for c in self.env.containerlist]
		self.schedulerinfo.append(schedulerinfo)

	def saveStats(self, deployed, migrated, destroyed, newcontainers, selectedcontainers, decision):	
		self.saveHostInfo()
		self.saveWorkloadInfo(deployed, migrated, newcontainers)
		self.saveContainerInfo()
		self.saveAllContainerInfo()
		self.saveMetrics(destroyed)
