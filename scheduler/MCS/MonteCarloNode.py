from math import sqrt, log
from copy import deepcopy
import random
from .SimpleObjects import *

c1 = 0.1
c2 = 0.1

class MonteCarloNode():
	def __init__(self, parent, decision = None):
		self.v = 0
		self.n = 1
		self.children = []
		self.parent = parent
		self.decision = decision
		if parent is not None: 
			self.senv = deepcopy(parent.senv)
			self.v = self.senv.getScore()
			if decision: self.senv.execDecision(decision)

	def select(self):
		curnode = self
		while curnode.children:
			model_scores = [child.v for child in curnode.children]
			scores = [child.v + sqrt(c1 * log(curnode.n) / (child.n + 1e-5)) for child in curnode.children]
			scores[np.argmax(model_scores)] += c2
			curnode = curnode.children[np.argmax(scores)]
		return curnode

	def expand(self, expansionSize, containerIDs):
		assert not self.children
		decisions = []
		cur_alloc = self.senv.getAllocDict()
		containers = list(set(list(cur_alloc.keys())))
		chosen_containers = random.choices(containers+containerIDs, k=min(len(containers+containerIDs), expansionSize))
		chosen_containers = containerIDs+chosen_containers
		curnode = self
		while curnode.parent: 
			chosen_containers = list(filter(lambda a: a != curnode.decision[0], chosen_containers))
			curnode = curnode.parent
		for cid in chosen_containers:
			hosts_done = [d[1] for d in decisions]
			hosts_av = list(set(range(len(self.senv.hostlist))).difference(set(hosts_done))) 
			random.shuffle(hosts_av)
			for hid in hosts_av:
				if self.senv.checkDecision((cid, hid)):
					decisions.append((cid, hid))
					break
		for decision in decisions:
			self.children.append(MonteCarloNode(self, decision))

	def backprop(self):
		curnode = self
		sumv, branchlen = 0, 0
		while curnode.parent is not None:
			sumv += curnode.v; branchlen += 1
			curnode = curnode.parent
		newv = sumv / (branchlen + 1e-5)
		curnode = self
		while curnode.parent is not None:
			curnode.v = newv; curnode.n += 1
			curnode = curnode.parent

	def getstr(self):
		if not self.children: return str(self)
		string = ''
		for i, child in enumerate(self.children):
			string = string + child.getstr()
			string += (',' if i < len(self.children) -1 else '')
		return f'('+string+f'){str(self)}'

	def __str__(self):
		if self.decision:
			return f'{self.decision[0]}^{self.decision[1]}^'+'{'+"{:.3f}".format(self.v)+f'^{self.n}'+'}'
		return '{'+"{:.3f}".format(self.v)+f'^{self.n}'+'}'
