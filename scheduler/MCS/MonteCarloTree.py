from .MonteCarloNode import *
from ete3 import Tree

class MonteCarloTree():
	def __init__(self, expansionSize):
		self.root = MonteCarloNode(None)
		self.expansionSize = expansionSize

	def setEnv(self, senv):
		self.root.senv = senv
		self.root.v = senv.getScore()

	def cumulateDecisions(self):
		decisions = []
		curnode = self.root
		while curnode.children:
			model_scores = [child.v for child in curnode.children]
			best_child = curnode.children[np.argmax(model_scores)]
			decision = best_child.decision
			curnode = best_child
			decisions.append(decision)
		return decisions

	def visualize(self):
		t = Tree(f'({self.root.getstr()});', format=1)
		print(t.get_ascii(show_internal=True))

	def checkDecisions(self, containerIDs, decisions):
		cids = [decision[0] for decision in decisions]
		for c in containerIDs:
			if c not in cids: return False
		return True

	def runSimulations(self, numSim, containerIDs):
		print(containerIDs)
		for _ in range(numSim):
			leafnode = self.root.select()
			leafnode.expand(self.expansionSize, containerIDs)
			leafnode.backprop()
		# self.visualize()
		decisions = self.cumulateDecisions()
		for cid in containerIDs:
			if cid not in [decision[0] for decision in decisions]:
				decisions.append((cid, random.choice(range(len(self.root.senv.hostlist)))))
		print(decisions)
		return decisions

