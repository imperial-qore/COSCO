import numpy as np
import torch

class SimpleEnv():
	def __init__(self, env, apps, graph, model, max_container_ips):
		self.containerlist = [SimpleContainer(c.getApparentIPS() if c else 0) for c in env.containerlist]
		self.hostlist = []
		for hostID, host in enumerate(env.hostlist):
			clist = [self.containerlist[i] for i in env.getContainersOfHost(hostID)]
			self.hostlist.append(SimpleHost(max_container_ips, clist))
		self.max_container_ips = max_container_ips
		self.model = model
		self.apps, self.graph = apps, graph

	def getScore(self):
		init = self.getInit()
		return 1 - self.model(init, self.apps, self.graph)

	def getAlloc(self):
		alloc = []
		for cid, container in enumerate(self.containerlist):
			oneHot = [0] * len(self.hostlist)
			hid = self.getHostid(cid)
			if hid != -1: oneHot[hid] = 1
			else: oneHot[np.random.randint(0,len(self.hostlist))] = 1
			alloc.append(oneHot)
		return alloc

	def getCPU(self):
		cpu = [host.getCPU() for host in self.hostlist]
		cpu = np.array([cpu]).transpose()
		cpuC = [c.ips/self.max_container_ips for c in self.containerlist]
		cpuC = np.array([cpuC]).transpose()
		cpu = np.concatenate((cpu, cpuC), axis=1)
		return cpu

	def getInit(self):
		cpu, alloc = self.getCPU(), self.getAlloc()
		init = np.concatenate((cpu, alloc), axis=1)
		init = torch.tensor(init, dtype=torch.float)
		return init

	def getHostid(self, cid):
		for hid, host in enumerate(self.hostlist):
			if cid in host.containers:
				return hid
			return -1

	def getAllocDict(self):
		cur_alloc = {}
		for cid, c in enumerate(self.containerlist):
			hid = self.getHostid(cid)
			if hid != -1: cur_alloc[cid] = hid
		return cur_alloc

	def getDecisionFromChild(self, senv):
		prev_alloc = self.getAllocDict()
		new_alloc = senv.getAllocDict()
		decision = []
		for cid in prev_alloc:
			new_host = new_alloc[cid]
			if prev_alloc[cid] != new_host: decision.append((cid, new_host))
		return decision

	def checkDecision(self, decision):
		cid, hid = decision
		hostips = np.sum([c.ips for c in self.hostlist[hid].containers])
		return self.containerlist[cid].ips + hostips <= self.hostlist[hid].ipscapacity

	def execDecision(self, decision):
		cid, hid = decision
		curhid = self.getHostid(cid)
		if cid in self.hostlist[curhid].containers:
			self.hostlist[curhid].containers.remove(self.containerlist[cid])
		self.hostlist[hid].containers.append(self.containerlist[cid])

class SimpleContainer():
	def __init__(self, ips):
		self.ips = ips

class SimpleHost():
	def __init__(self, ipscapacity, containers):
		self.containers = containers
		self.ipscapacity = ipscapacity

	def getCPU(self):
		ips = np.sum([c.ips for c in self.containers])
		return ips / self.ipscapacity