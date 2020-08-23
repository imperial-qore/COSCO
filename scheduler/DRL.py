from .A2C.rl import *
from .Scheduler import *

import pickle
from os import path, mkdir
from random import sample

class DRLScheduler(Scheduler):
	def __init__(self, data_type):
		super().__init__()
		self.model = eval(data_type+"_RL()")
		self.last_schedule = None
		self.model, self.optimizer, self.epoch, self.accuracy_list = load_model(data_type+'_RL', self.model, data_type)
		self.data_type = data_type
		self.buffer = []
		directory = 'scheduler/A2C/dataset/'
		self.buffer_path = directory + 'buffer.pkl'
		if not path.exists(directory): mkdir(directory)
		if path.exists(self.buffer_path):
			with open(self.buffer_path, "rb") as handle:
				self.buffer = pickle.load(handle)

	def get_current_schedule(self):
		cpu = [host.getCPU()/100 for host in self.env.hostlist]
		cpu = np.array([cpu]).transpose()
		if '_' in self.data_type:
			cpuC = [(c.getApparentIPS()/6000 if c else 0) for c in self.env.containerlist]
			cpuC = np.array([cpuC]).transpose()
			cpu = np.concatenate((cpu, cpuC), axis=1)
		alloc = []; prev_alloc = {}
		for c in self.env.containerlist:
			oneHot = [0] * len(self.env.hostlist)
			if c: prev_alloc[c.id] = c.getHostID()
			if c and c.getHostID() != -1: oneHot[c.getHostID()] = 1
			else: oneHot[np.random.randint(0,len(self.env.hostlist))] = 1
			alloc.append(oneHot)
		init = np.concatenate((cpu, alloc), axis=1)
		init = torch.tensor(init, dtype=torch.float)
		return init, prev_alloc

	def get_last_value(self):
		energy = self.env.stats.metrics[-1]['energytotalinterval']
		energy = (energy - 9800)/9000
		if '_' in self.data_type:
			latency = self.env.stats.metrics[-1]['avgresponsetime']
			latency = (latency)/7000
			return Coeff_Energy*energy + Coeff_Latency*latency
		return energy

	def run_DRL(self):
		self.epoch += 1
		schedule_next, prev_alloc = self.get_current_schedule()
		value_t = self.get_last_value() if self.last_schedule != None else 0
		schedule_t = self.last_schedule if self.last_schedule != None else schedule_next
		if len(self.buffer) >= 10:
			for replay in sample(self.buffer, 10):
				backprop(replay[0], replay[1], replay[2], replay[3], self.optimizer)
		vl, pl, action = backprop(self.model, schedule_t, value_t, schedule_next, self.optimizer)
		self.buffer.append((self.model, schedule_t, value_t, schedule_next))
		self.buffer = self.buffer[-50:]
		self.accuracy_list.append((vl+pl, vl, pl))
		print(vl, pl)
		decision = []
		for cid in prev_alloc:
			one_hot = action.tolist()
			new_host = one_hot.index(max(one_hot))
			if prev_alloc[cid] != new_host: decision.append((cid, new_host))
		self.last_schedule = schedule_t
		if self.epoch % 10 == 0:
			save_model(self.model, self.optimizer, self.epoch, self.accuracy_list)
			with open(self.buffer_path, "wb") as handle:
				pickle.dump(self.buffer, handle)
			plot_accuracies(self.accuracy_list)
		return decision

	def selection(self):
		return []

	def placement(self, containerIDs):
		first_alloc = np.all([not (c and c.getHostID() != -1) for c in self.env.containerlist])
		decision = self.run_DRL()
		return decision